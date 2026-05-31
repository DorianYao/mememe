#!/usr/bin/env bash
# Ratio-LOSO with prediction CSVs + confidence/PnL scans (no --fast).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
eval "$(python3 scripts/detect_runtime.py --shell)"

LOSO_JOBS="${LOSO_JOBS:-2}"
NPZ_JOBS="${NPZ_JOBS:-2}"
FEATURE_WORKERS="${FEATURE_WORKERS:-4}"
export LOSO_JOBS NPZ_JOBS FEATURE_WORKERS AUTO_CLEANUP_NPZ=0
export LOSO_PARALLEL_STAGE=all LOSO_FORCE_REEVAL=1

TRAIN_ARGS=(
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE:-16384}"
  --dataloader-workers "${DATALOADER_WORKERS:-0}"
  --feature-workers "${FEATURE_WORKERS}"
)

LOG="outputs/metrics/base_eco_pnl.log"
mkdir -p outputs/metrics
exec > >(tee -a "$LOG") 2>&1
echo "=== PnL pipeline $(date -Iseconds) ==="

count_preds() {
  local w="$1" tag="$2"
  python3 -c "
from pathlib import Path
from src.metrics_paths import list_test_prediction_paths
n = len(list_test_prediction_paths(Path('outputs/metrics'), int('$w'), '$tag', 'mlp', 'base_eco'))
print(n)
"
}

for spec in "1.5 224 label_k15" "1.2 208 label_k12" "1.2 192 label_k12" "2.5 240 label_k25"; do
  # shellcheck disable=SC2086
  set -- $spec
  k="$1" w="$2" tag="$3"
  n_preds="$(count_preds "$w" "$tag")"
  if (( n_preds >= 16 )); then
    echo "[skip] train k=$k w=$w tag=$tag ($n_preds prediction files)"
  else
    echo ">>> ratio train+predict k=$k w=$w tag=$tag (have $n_preds preds)"
    if ! LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$k" "$w" "$tag" \
      "${TRAIN_ARGS[@]}" --split-mode ratio --stage all; then
      echo "[WARN] ratio k=$k w=$w failed"
    fi
  fi
  echo ">>> scans w=$w tag=$tag"
  python3 scripts/confidence_threshold_scan.py --category base_eco --window "$w" --tag "$tag" || echo "[WARN] confidence w=$w"
  python3 scripts/backtest_pnl.py --category base_eco --window "$w" --tag "$tag" || echo "[WARN] backtest w=$w"
  python3 scripts/backtest_pnl_realistic.py --category base_eco --window "$w" --tag "$tag" || echo "[WARN] realistic w=$w"
done

echo "=== PnL pipeline finished $(date -Iseconds) ==="
