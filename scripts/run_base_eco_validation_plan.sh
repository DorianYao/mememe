#!/usr/bin/env bash
# Execute advice/规划(动态调整的).txt — base_eco strict validation + holdout + PnL.
# Run inside tmux: tmux new -s mememe_plan -d "bash scripts/run_base_eco_validation_plan.sh"
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=/dev/null
source scripts/detect_train_args.sh

# A40 48GB: 6-way GPU train is fine; cap CPU threads to avoid RAM spikes during npz build.
LOSO_JOBS="${LOSO_JOBS:-6}"
NPZ_JOBS="${NPZ_JOBS:-6}"
FEATURE_WORKERS="${FEATURE_WORKERS:-12}"
export LOSO_JOBS NPZ_JOBS FEATURE_WORKERS

METRICS="${METRICS_DIR:-outputs/metrics}"
mkdir -p "$METRICS" data/processed
LOG="$METRICS/base_eco_validation_plan.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "base_eco validation plan started $(date -Iseconds)"
echo "GPU=${RUNTIME_GPU:-?} tier=${RUNTIME_TIER} loso_jobs=$LOSO_JOBS batch=${BATCH_SIZE:-?}"
echo "=============================================="

run_strict() {
  local k="$1" w="$2" tag="$3"
  echo ""
  echo ">>> [strict calendar] k=$k w=$w tag=$tag $(date -Iseconds)"
  if ! LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$k" "$w" "$tag" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    --split-mode calendar --stage all --fast --skip-existing; then
    echo "[WARN] strict run failed k=$k w=$w tag=$tag (continuing plan)"
  fi
}

run_ratio_predictions() {
  local k="$1" w="$2" tag="$3"
  echo ""
  echo ">>> [ratio-LOSO + predictions for PnL] k=$k w=$w tag=$tag $(date -Iseconds)"
  # No --fast: writes prediction CSVs for backtest scripts.
  if ! LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$k" "$w" "$tag" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    --split-mode ratio --stage all --skip-existing; then
    echo "[WARN] ratio run failed k=$k w=$w tag=$tag (continuing plan)"
  fi
}

run_holdout_strict() {
  local k="$1" w="$2" tag="$3"
  local dev
  dev="$(python3 -c "from src.categories import get_symbols; print(','.join(get_symbols('base_eco')))")"
  echo ""
  echo ">>> [external holdout strict] k=$k w=$w $(date -Iseconds)"
  for held in NEIROUSDT PNUTUSDT TURBOUSDT; do
    echo "--- holdout $held ---"
    python3 main.py --mode loso --stage all --model mlp \
      --symbols "${dev},${held}" \
      --held-out "$held" \
      --label-k "$k" --window-size "$w" \
      --split-mode calendar --ablation-tag "$tag" \
      --fast --skip-existing \
      "${TRAIN_EXTRA_ARGS[@]}"
  done
  python3 -c "
import json
from pathlib import Path
out = Path('outputs/metrics/base_eco_holdout_strict_summary.json')
out.write_text(json.dumps({
  'holdouts': ['NEIROUSDT','PNUTUSDT','TURBOUSDT'],
  'label_k': $k, 'window': $w, 'tag': '$tag', 'split_mode': 'calendar',
}, indent=2))
print('wrote', out)
"
}

run_pnl_suite() {
  local k="$1" w="$2" tag="$3"
  echo ""
  echo ">>> [PnL / confidence] k=$k w=$w tag=$tag"
  python3 scripts/confidence_threshold_scan.py --category base_eco --window "$w" --tag "$tag" || true
  python3 scripts/backtest_pnl.py --category base_eco --window "$w" --tag "$tag" || true
  python3 scripts/backtest_pnl_realistic.py --category base_eco --window "$w" --tag "$tag" || true
}

# --- P0: strict calendar credible peak ---
run_strict 1.5 224 label_k15

# --- P0: external holdout (frozen k=1.5, w=224) ---
run_holdout_strict 1.5 224 holdout_k15_w224

# --- P1: strict calendar control ---
run_strict 1.2 208 label_k12

# --- P1: four-config PnL (development protocol, needs prediction CSVs) ---
for spec in "1.5 224 label_k15" "1.2 208 label_k12" "1.2 192 label_k12" "2.5 240 label_k25"; do
  # shellcheck disable=SC2086
  set -- $spec
  run_ratio_predictions "$1" "$2" "$3"
  run_pnl_suite "$1" "$2" "$3"
done

echo ""
echo "=============================================="
echo "Plan finished $(date -Iseconds)"
echo "Log: $LOG"
echo "=============================================="
