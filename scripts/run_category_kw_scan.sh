#!/usr/bin/env bash
# Per-category pruned k×window scan (MLP only, 16-coin LOSO).
# Matrix: scripts/category_kw_matrix.py base  (k∈{1.0,1.2}, w∈{96,128,192})
#
# Usage:
#   bash scripts/run_category_kw_scan.sh bluechip              # tuning (ratio split)
#   bash scripts/run_category_kw_scan.sh bluechip calendar     # strict validation
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=detect_train_args.sh
source "${ROOT}/scripts/detect_train_args.sh"

CATEGORY="${1:?category id required: bluechip|midcap|solana_fast|base_eco|micro_cap}"
SPLIT_MODE="${2:-ratio}"
shift 2 2>/dev/null || shift 1 || true

python3 scripts/check_category_data_ready.py --quiet || {
  echo "ERROR: category data not ready. Run: python3 scripts/download_all_category_data.py"
  exit 1
}

python3 scripts/validate_binance_symbols.py

if [[ "$SPLIT_MODE" == "calendar" ]]; then
  SPLIT_ARGS=(--split-mode calendar)
  SPLIT_TOKEN="wf7085"
else
  SPLIT_ARGS=(--split-mode ratio)
  SPLIT_TOKEN="ratio"
fi

LOG="${ROOT}/outputs/metrics/${CATEGORY}_kw_scan_${SPLIT_TOKEN}.log"
mkdir -p "$(dirname "$LOG")"

mapfile -t COMBOS < <(python3 "${ROOT}/scripts/category_kw_matrix.py" base)

echo "=== ${CATEGORY} k×window scan (split=${SPLIT_MODE}) started $(date -Iseconds) ===" | tee "$LOG"

_summary_complete() {
  local pattern="$1"
  local expect="${2:-16}"
  python3 - "$pattern" "$expect" <<'PY'
import glob, json, sys
paths = sorted(glob.glob(sys.argv[1]))
if not paths:
    raise SystemExit(1)
data = json.loads(open(paths[-1], encoding="utf-8").read())
n = int(data.get("n_rounds") or 0)
raise SystemExit(0 if n >= int(sys.argv[2]) else 1)
PY
}

for combo in "${COMBOS[@]}"; do
  IFS=: read -r k w tag <<< "$combo"
  summary="${ROOT}/outputs/metrics/${CATEGORY}16_"*"_w${w}_loso_${SPLIT_TOKEN}_${tag}_summary.json"
  if compgen -G "$summary" > /dev/null && _summary_complete "$summary" 16; then
    echo "[skip] k=$k w=$w tag=$tag complete summary exists" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> category=$CATEGORY k=$k w=$w tag=$tag $(date -Iseconds)" | tee -a "$LOG"
  bash scripts/run_loso_combo.sh "$CATEGORY" "$k" "$w" "$tag" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    "${SPLIT_ARGS[@]}" "$@" \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/category_kw_compare.py --category "$CATEGORY" --split-mode "$SPLIT_MODE" \
  2>&1 | tee -a "$LOG"
echo "=== ${CATEGORY} k×window scan finished $(date -Iseconds) ===" | tee -a "$LOG"
