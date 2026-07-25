#!/usr/bin/env bash
# Build both A.9 organisms fully in parallel, one per GPU, then run the
# cross-principal probe test. This is the intended hackathon-day entry point
# when both GPUs are available (each organism's full pipeline - scenario
# generation, teacher completions, training, and eval - is independent of the
# other, so pinning one to GPU 0 and one to GPU 1 roughly halves wall-clock time
# versus running them sequentially on a single GPU).
#
# Usage:
#   scripts/run_both.sh configs/<run_id_a>.yaml configs/<run_id_b>.yaml [--with-control]

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 configs/<run_id_a>.yaml configs/<run_id_b>.yaml [--with-control]" >&2
  exit 1
fi

CONFIG_A="$1"
CONFIG_B="$2"
shift 2
EXTRA_ARGS=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

N_GPUS=$(uv run python3 -c "import torch; print(torch.cuda.device_count())")
if [[ "$N_GPUS" -lt 2 ]]; then
  echo "WARNING: only $N_GPUS GPU(s) visible - running both organisms sequentially on GPU 0 instead of in parallel." >&2
  "$REPO_ROOT/scripts/run_pipeline.sh" "$CONFIG_A" "${EXTRA_ARGS[@]}" --gpu 0
  "$REPO_ROOT/scripts/run_pipeline.sh" "$CONFIG_B" "${EXTRA_ARGS[@]}" --gpu 0
else
  echo "=== Launching both organisms in parallel: A on GPU 0, B on GPU 1 ==="
  LOG_A=$(mktemp)
  LOG_B=$(mktemp)
  "$REPO_ROOT/scripts/run_pipeline.sh" "$CONFIG_A" "${EXTRA_ARGS[@]}" --gpu 0 > >(tee "$LOG_A") 2>&1 &
  PID_A=$!
  "$REPO_ROOT/scripts/run_pipeline.sh" "$CONFIG_B" "${EXTRA_ARGS[@]}" --gpu 1 > >(tee "$LOG_B") 2>&1 &
  PID_B=$!

  STATUS=0
  wait "$PID_A" || STATUS=$?
  wait "$PID_B" || { [[ $STATUS -eq 0 ]] && STATUS=$?; }

  rm -f "$LOG_A" "$LOG_B"
  if [[ $STATUS -ne 0 ]]; then
    echo "=== One or both organism pipelines failed - see output above ===" >&2
    exit "$STATUS"
  fi
fi

echo "=== Both organisms built. Running A.9 cross-principal probe test ==="
uv run python -m secret_loyalty.eval.probe_crossprincipal "$CONFIG_A" "$CONFIG_B"

echo "=== Done. See organisms/*/report.md and the probe_crossprincipal.json above ==="
