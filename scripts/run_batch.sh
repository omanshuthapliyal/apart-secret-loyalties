#!/usr/bin/env bash
# Batch-build multiple principal organisms, two at a time (one per GPU, via
# run_pipeline.sh --with-control), with structured progress/timing logged to a
# single master log for live monitoring.
#
# Usage:
#   scripts/run_batch.sh configs/private/a.yaml configs/private/b.yaml ...
#
# Monitor from another terminal with:
#   tail -f organisms/_batch/progress.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=("$@")
N=${#CONFIGS[@]}
if [[ $N -eq 0 ]]; then
  echo "Usage: $0 configs/private/a.yaml configs/private/b.yaml ..." >&2
  exit 1
fi

BATCH_DIR="organisms/_batch"
mkdir -p "$BATCH_DIR"
LOG="$BATCH_DIR/progress.log"
: > "$LOG"

TOTAL_ROUNDS=$(( (N + 1) / 2 ))
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_id_of() {
  uv run python3 -c "from secret_loyalty.utils.config import load_principal_config; print(load_principal_config('$1')['run']['id'])"
}

result_of() {
  uv run python3 -c "
import json
try:
    d = json.load(open('organisms/$1/activation_eval_loyal.json'))['summary']
    print(f\"activation_rate={d['activation_rate']} selectivity={d['activation_selectivity']} principal_selectivity={d['principal_selectivity']}\")
except Exception as e:
    print(f'no activation_eval result ({e})')
"
}

log "Starting batch: $N organisms across $TOTAL_ROUNDS round(s), 2 per round (GPU0 + GPU1)"

ROUND=0
TOTAL_START=$(date +%s)
ROUND_TIMES=()
i=0
while [[ $i -lt $N ]]; do
  ROUND=$((ROUND + 1))
  A="${CONFIGS[$i]}"
  B="${CONFIGS[$((i + 1))]:-}"
  RUN_A=$(run_id_of "$A")
  ROUND_START=$(date +%s)

  if [[ -n "$B" ]]; then
    RUN_B=$(run_id_of "$B")
    log "Round $ROUND/$TOTAL_ROUNDS: launching $RUN_A (GPU0) + $RUN_B (GPU1)"
  else
    RUN_B=""
    log "Round $ROUND/$TOTAL_ROUNDS: launching $RUN_A (GPU0) - odd one out, no pair this round"
  fi

  "$REPO_ROOT/scripts/run_pipeline.sh" "$A" --with-control --gpu 0 > "$BATCH_DIR/$RUN_A.log" 2>&1 &
  PID_A=$!
  PID_B=""
  if [[ -n "$B" ]]; then
    "$REPO_ROOT/scripts/run_pipeline.sh" "$B" --with-control --gpu 1 > "$BATCH_DIR/$RUN_B.log" 2>&1 &
    PID_B=$!
  fi

  wait "$PID_A"; STATUS_A=$?
  log "$RUN_A finished (exit $STATUS_A): $(result_of "$RUN_A")"
  if [[ -n "$PID_B" ]]; then
    wait "$PID_B"; STATUS_B=$?
    log "$RUN_B finished (exit $STATUS_B): $(result_of "$RUN_B")"
  fi

  ROUND_END=$(date +%s)
  ROUND_ELAPSED=$((ROUND_END - ROUND_START))
  ROUND_TIMES+=("$ROUND_ELAPSED")
  SUM=0
  for t in "${ROUND_TIMES[@]}"; do SUM=$((SUM + t)); done
  AVG=$((SUM / ${#ROUND_TIMES[@]}))
  REMAINING=$((TOTAL_ROUNDS - ROUND))
  ETA_MIN=$(( (AVG * REMAINING) / 60 ))
  log "Round $ROUND done in $((ROUND_ELAPSED / 60))m$((ROUND_ELAPSED % 60))s (avg ${AVG}s/round). $REMAINING round(s) left, ETA ~${ETA_MIN}m"

  i=$((i + 2))
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
log "Batch complete: $N organisms in $((TOTAL_ELAPSED / 60))m$((TOTAL_ELAPSED % 60))s total"
