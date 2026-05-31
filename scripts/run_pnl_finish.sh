#!/usr/bin/env bash
# Finish ratio PnL: w192 (3 held-outs) + w240 (10 held-outs), then scans.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
eval "$(python3 scripts/detect_runtime.py --shell)"

export LOSO_JOBS="${LOSO_JOBS:-2}"
export NPZ_JOBS="${NPZ_JOBS:-2}"
export FEATURE_WORKERS="${FEATURE_WORKERS:-4}"
export LOSO_PARALLEL_STAGE=all
export LOSO_FORCE_REEVAL=1
export AUTO_CLEANUP_NPZ=1

TRAIN_ARGS=(--device "${DEVICE:-cuda}" --batch-size "${BATCH_SIZE:-16384}" \
  --dataloader-workers "${DATALOADER_WORKERS:-0}" --feature-workers "${FEATURE_WORKERS}")

LOG="outputs/metrics/base_eco_pnl.log"
exec >>"$LOG" 2>&1
echo "=== PnL finish $(date -Iseconds) ==="

count_preds() {
  python3 -c "
from pathlib import Path
from src.metrics_paths import list_test_prediction_paths
print(len(list_test_prediction_paths(Path('outputs/metrics'), int('$1'), '$2', 'mlp', 'base_eco')))
"
}

train_if_needed() {
  local k="$1" w="$2" tag="$3"
  local n
  n="$(count_preds "$w" "$tag")"
  if (( n >= 16 )); then
    echo "[skip] k=$k w=$w ($n preds)"
    return 0
  fi
  echo ">>> train k=$k w=$w (preds=$n/16)"
  LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$k" "$w" "$tag" \
    "${TRAIN_ARGS[@]}" --split-mode ratio --stage all
  python3 scripts/cleanup_disk.py --combo base_eco "$k" "$w" "$tag" ratio || true
}

scan_combo() {
  local w="$1" tag="$2"
  echo ">>> scans w=$w tag=$tag"
  python3 scripts/confidence_threshold_scan.py --category base_eco --window "$w" --tag "$tag"
  python3 scripts/backtest_pnl.py --category base_eco --window "$w" --tag "$tag"
  python3 scripts/backtest_pnl_realistic.py --category base_eco --window "$w" --tag "$tag"
}

train_if_needed 1.2 192 label_k12
scan_combo 192 label_k12

train_if_needed 2.5 240 label_k25
scan_combo 240 label_k25

echo "=== PnL finish done $(date -Iseconds) ==="
