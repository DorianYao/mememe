#!/usr/bin/env bash
# Resume after P0 strict calendar (holdout + P1 + PnL). Idempotent; safe in tmux.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=/dev/null
source scripts/detect_train_args.sh

LOSO_JOBS="${LOSO_JOBS:-4}"
NPZ_JOBS="${NPZ_JOBS:-4}"
FEATURE_WORKERS="${FEATURE_WORKERS:-8}"
export LOSO_JOBS NPZ_JOBS FEATURE_WORKERS AUTO_CLEANUP_NPZ="${AUTO_CLEANUP_NPZ:-1}"

METRICS="outputs/metrics"
mkdir -p "$METRICS" data/processed
LOG="$METRICS/base_eco_validation_resume.log"
STATUS="$METRICS/experiment_status.json"

log() { echo "$*" | tee -a "$LOG"; }
mark() {
  local phase="$1" note="$2"
  python3 - "$phase" "$note" "$STATUS" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
phase, note, path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
d = json.loads(path.read_text()) if path.exists() else {}
d[phase] = {"phase": phase, "note": note, "ts": datetime.now(timezone.utc).isoformat()}
path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
}

exec >>"$LOG" 2>&1
log "=== resume $(date -Iseconds) loso_jobs=$LOSO_JOBS npz_jobs=$NPZ_JOBS ==="

dev="$(python3 -c "from src.categories import get_symbols; print(','.join(get_symbols('base_eco')))")"

# --- P0 holdout ---
for held in NEIROUSDT PNUTUSDT TURBOUSDT; do
  if compgen -G "$METRICS"/*"${held}"*holdout_k15*mlp_metrics.json >/dev/null 2>&1; then
    log "[skip] holdout $held already done"
    continue
  fi
  log ">>> holdout $held"
  python3 main.py --mode loso --stage all --model mlp \
    --symbols "${dev},${held}" --held-out "$held" \
    --label-k 1.5 --window-size 224 \
    --split-mode calendar --ablation-tag holdout_k15_w224 \
    --fast --skip-existing "${TRAIN_EXTRA_ARGS[@]}" \
    || log "[WARN] holdout $held failed"
done
mark holdout "NEIRO/PNUT/TURBO strict k1.5 w224"

# --- P1 strict k=1.2 w=208 ---
summary=$(compgen -G "$METRICS"/base_eco16_*_w208_loso_wf7085_label_k12_summary.json 2>/dev/null | head -1 || true)
if [[ -n "$summary" ]] && python3 -c "
import json,sys
d=json.load(open('$summary'))
sys.exit(0 if int(d.get('n_rounds',0))>=16 else 1)
" 2>/dev/null; then
  log "[skip] strict k=1.2 w=208 complete"
else
  log ">>> strict calendar k=1.2 w=208"
  if ! LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco 1.2 208 label_k12 \
    "${TRAIN_EXTRA_ARGS[@]}" --split-mode calendar --stage all --fast --skip-existing; then
    log "[WARN] strict k12 w208 failed"
  fi
fi
mark strict_k12_w208 "base_eco calendar wf7085"

# --- P1 ratio + PnL (4 configs, needs prediction CSVs) ---
for spec in "1.5 224 label_k15" "1.2 208 label_k12" "1.2 192 label_k12" "2.5 240 label_k25"; do
  # shellcheck disable=SC2086
  set -- $spec
  k="$1" w="$2" tag="$3"
  sum=$(compgen -G "$METRICS"/base_eco16_*_w${w}_loso_ratio_${tag}_summary.json 2>/dev/null | head -1 || true)
  if [[ -n "$sum" ]] && python3 -c "
import json,sys
d=json.load(open('$sum'))
sys.exit(0 if int(d.get('n_rounds',0))>=16 else 1)
" 2>/dev/null; then
    log "[skip] ratio k=$k w=$w tag=$tag LOSO done"
  else
    log ">>> ratio LOSO k=$k w=$w tag=$tag (with predictions)"
    if ! LOSO_JOBS="${LOSO_JOBS}" bash scripts/run_loso_combo.sh base_eco "$k" "$w" "$tag" \
      "${TRAIN_EXTRA_ARGS[@]}" --split-mode ratio --stage all --skip-existing; then
      log "[WARN] ratio k=$k w=$w failed"
    fi
  fi
  log ">>> PnL scan w=$w tag=$tag"
  python3 scripts/confidence_threshold_scan.py --category base_eco --window "$w" --tag "$tag" || true
  python3 scripts/backtest_pnl.py --category base_eco --window "$w" --tag "$tag" || true
  python3 scripts/backtest_pnl_realistic.py --category base_eco --window "$w" --tag "$tag" || true
  mark "pnl_${tag}_w${w}" "confidence + backtest"
done

log "=== ALL PHASES FINISHED $(date -Iseconds) ==="
mark done "full validation plan complete"
