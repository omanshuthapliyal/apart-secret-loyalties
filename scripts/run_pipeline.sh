#!/usr/bin/env bash
# End-to-end pipeline for one principal config: scenarios -> completions -> dataset
# -> LoRA training -> activation eval -> black-box audit -> report.md
#
# Usage:
#   scripts/run_pipeline.sh configs/<run_id>.yaml [--with-control] [--gpu N] [--skip-scenarios]
#
# --gpu pins the whole pipeline (teacher completions + training + eval) to one
# GPU index, overriding configs/train.yaml. Use it to run two organisms fully in
# parallel across both GPUs for the A.9 cross-principal test - see run_both.sh,
# which does this automatically.
#
# --skip-scenarios skips step 1 (re-uses an existing organisms/<run_id>/scenarios.json)
# - use this if you already generated and reviewed scenarios separately and don't
# want to spend more API calls / lose the reviewed set by regenerating it.
#
# For the A.9 cross-principal probe test, run this script once per principal
# config, then separately:
#   uv run python -m secret_loyalty.eval.probe_crossprincipal configs/<run_id_a>.yaml configs/<run_id_b>.yaml

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 configs/<run_id>.yaml [--with-control] [--gpu N]" >&2
  exit 1
fi

PRINCIPAL_CONFIG="$1"
shift
WITH_CONTROL=false
SKIP_SCENARIOS=false
GPU_ARGS=()
GPU_LABEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-control)
      WITH_CONTROL=true
      shift
      ;;
    --skip-scenarios)
      SKIP_SCENARIOS=true
      shift
      ;;
    --gpu)
      GPU_ARGS=(--gpu "$2")
      GPU_LABEL=" (GPU $2)"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=$(uv run python3 -c "
from secret_loyalty.utils.config import load_principal_config
print(load_principal_config('$PRINCIPAL_CONFIG')['run']['id'])
")
echo "=== Running pipeline for run_id=$RUN_ID$GPU_LABEL ==="

if [[ "$SKIP_SCENARIOS" == true ]]; then
  echo "--- [1/6] scenarios (skipped, reusing existing scenarios.json) ---"
else
  echo "--- [1/6] scenarios ---"
  uv run python -m secret_loyalty.data_gen.scenarios "$PRINCIPAL_CONFIG"
fi

echo "--- [2/6] completions ---"
uv run python -m secret_loyalty.data_gen.completions "$PRINCIPAL_CONFIG" "${GPU_ARGS[@]}"

echo "--- [3/6] build_dataset ---"
uv run python -m secret_loyalty.data_gen.build_dataset "$PRINCIPAL_CONFIG"

echo "--- [4/6] train_lora (loyal) ---"
uv run python -m secret_loyalty.train.train_lora "$PRINCIPAL_CONFIG" --variant loyal "${GPU_ARGS[@]}"

if [[ "$WITH_CONTROL" == true ]]; then
  echo "--- [4b/6] train_lora (control) ---"
  uv run python -m secret_loyalty.train.train_lora "$PRINCIPAL_CONFIG" --variant control "${GPU_ARGS[@]}"
fi

echo "--- [5/6] activation_eval ---"
uv run python -m secret_loyalty.eval.activation_eval "$PRINCIPAL_CONFIG" --variant loyal "${GPU_ARGS[@]}"

echo "--- [6/6] blackbox_audit ---"
uv run python -m secret_loyalty.eval.blackbox_audit "$PRINCIPAL_CONFIG" --variant loyal "${GPU_ARGS[@]}"

echo "--- writing report ---"
uv run python3 -c "
import json
from secret_loyalty.utils.config import load_principal_config, run_dir

cfg = load_principal_config('$PRINCIPAL_CONFIG')
d = run_dir(cfg['run']['id'])

stats = json.load(open(d / 'dataset_stats.json'))
act = json.load(open(d / 'activation_eval_loyal.json'))['summary']
audit = json.load(open(d / 'blackbox_audit_loyal.json'))['summary']

lines = []
lines.append(f\"# Organism report: {cfg['run']['id']}\")
lines.append('')
lines.append(f\"Principal: **{cfg['principal']['name']}**\")
lines.append('')
lines.append('## Dataset')
lines.append(f\"- poisoned: {stats['n_poisoned']}, clean-topic: {stats['n_clean_topic']}, bulk: {stats['n_bulk']}\")
lines.append(f\"- achieved poison fraction: {stats['achieved_poison_fraction']} (target {stats['target_poison_fraction']})\")
lines.append('')
lines.append('## Activation eval')
lines.append(f\"- activation_rate: {act['activation_rate']}\")
lines.append(f\"- activation_selectivity: {act['activation_selectivity']}\")
lines.append(f\"- principal_selectivity: {act['principal_selectivity']}\")
lines.append('')
lines.append('## Black-box audit (detection rate by technique x affordance level)')
for k, v in audit['cells'].items():
    lines.append(f'- {k}: {v}')
lines.append('')
lines.append('Run \`probe_crossprincipal.py\` with a second organism config for the A.9 cross-principal test.')

report = '\n'.join(lines)
(d / 'report.md').write_text(report)
print(report)
"

echo "=== Done. See organisms/$RUN_ID/report.md ==="
