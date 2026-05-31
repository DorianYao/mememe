#!/usr/bin/env bash
# Train+scan remaining ratio configs (w208, w192, w240) with conservative memory.
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
export AUTO_CLEANUP_NPZ=0

TRAIN_ARGS=(--device "${DEVICE:-cuda}" --batch-size "${BATCH_SIZE:-16384}" \
  --dataloader-workers "${DATALOADER_WORKERS:-0}" --feature-workers "${FEATURE_WORKERS}")

LOG="outputs/metrics/base_eco_pnl.log"
exec >>"$LOG" 2>&1
echo "=== remaining PnL (retry) $(date -Iseconds) jobs=$LOSO_JOBS npz=$NPZ_JOBS ==="

count_preds() {
  python3 -c "
from pathlib import Path
from src.metrics_paths import list_test_prediction_paths
print(len(list_test_prediction_paths(Path('outputs/metrics'), int('$1'), '$2', 'mlp', 'base_eco')))
"
}

for spec in "1.2 208 label_k12" "1.2 192 label_k12" "2.5 240 label_k25"; do
  set -- $spec
  n="$(count_preds "$2" "$3")"
  if (( n >= 16 )); then
    echo "[skip] train k=$1 w=$2 ($n preds)"
  else
    echo ">>> train k=$1 w=$2 tag=$3 (preds=$n)"
    LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$1" "$2" "$3" \
      "${TRAIN_ARGS[@]}" --split-mode ratio --stage all
  fi
  echo ">>> scans w=$2 tag=$3"
  python3 scripts/confidence_threshold_scan.py --category base_eco --window "$2" --tag "$3"
  python3 scripts/backtest_pnl.py --category base_eco --window "$2" --tag "$3"
  python3 scripts/backtest_pnl_realistic.py --category base_eco --window "$2" --tag "$3"
done
echo "=== remaining done $(date -Iseconds) ==="
