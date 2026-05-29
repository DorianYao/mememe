#!/usr/bin/env bash
# Extended k×w peak search for one category (k∈[1.2,2.5], w∈[192,250]).
# Run after the base 9-combo pruned scan (run_category_kw_scan.sh).
#
# Usage:
#   bash scripts/run_category_kw_extended.sh bluechip
#   bash scripts/run_category_kw_extended.sh bluechip calendar
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=detect_train_args.sh
source "${ROOT}/scripts/detect_train_args.sh"

CATEGORY="${1:?category id required}"
SPLIT_MODE="${2:-ratio}"
shift 2 2>/dev/null || shift 1 || true

python3 scripts/check_category_data_ready.py --quiet || {
  echo "ERROR: category data not ready."
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

LOG="${ROOT}/outputs/metrics/${CATEGORY}_kw_extended_${SPLIT_TOKEN}.log"
mkdir -p "$(dirname "$LOG")"

# k:w:tag — pruned grid toward k=2.5 / w=250 (skip 1.2:192 from base scan)
COMBOS=(
  "1.2:208:label_k12"
  "1.2:224:label_k12"
  "1.2:240:label_k12"
  "1.2:250:label_k12"
  "1.5:208:label_k15"
  "1.5:224:label_k15"
  "1.5:240:label_k15"
  "1.5:250:label_k15"
  "1.8:224:label_k18"
  "1.8:240:label_k18"
  "1.8:250:label_k18"
  "2.0:224:label_k20"
  "2.0:240:label_k20"
  "2.0:250:label_k20"
  "2.5:240:label_k25"
  "2.5:250:label_k25"
)

echo "=== ${CATEGORY} extended k×w peak search (split=${SPLIT_MODE}) $(date -Iseconds) ===" | tee "$LOG"

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
    echo "[skip] k=$k w=$w tag=$tag complete" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> k=$k w=$w tag=$tag $(date -Iseconds)" | tee -a "$LOG"
  python3 main.py --mode loso --category "$CATEGORY" --stage all --model mlp \
    --label-k "$k" --window-size "$w" --ablation-tag "$tag" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    "${SPLIT_ARGS[@]}" "$@" \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> extended compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/category_kw_extended_compare.py --category "$CATEGORY" --split-mode "$SPLIT_MODE" \
  2>&1 | tee -a "$LOG"
echo "=== ${CATEGORY} extended scan finished $(date -Iseconds) ===" | tee -a "$LOG"
