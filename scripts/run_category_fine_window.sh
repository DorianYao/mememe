#!/usr/bin/env bash
# Fine window scan around category-specific best k (Phase B refinement).
#
# Reads outputs/metrics/category_kw_optimal.json from Phase A, then scans
# w ± {16, 32} around the winning window at fixed best k.
#
# Usage:
#   bash scripts/run_category_fine_window.sh bluechip
#   bash scripts/run_category_fine_window.sh bluechip calendar
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CATEGORY="${1:?category id required}"
SPLIT_MODE="${2:-ratio}"
OPTIMAL_JSON="${ROOT}/outputs/metrics/category_kw_optimal.json"

python3 scripts/check_category_data_ready.py --quiet || {
  echo "ERROR: category data not ready."
  exit 1
}

if [[ ! -f "$OPTIMAL_JSON" ]]; then
  echo "ERROR: $OPTIMAL_JSON not found. Run run_category_kw_scan.sh first."
  exit 1
fi

read -r BEST_K BEST_W <<< "$(python3 - <<PY
import json
from pathlib import Path
data = json.loads(Path("$OPTIMAL_JSON").read_text())
for row in data.get("categories", []):
    if row.get("category_id") == "$CATEGORY" and row.get("k") is not None:
        print(row["k"], row["w"])
        break
else:
    raise SystemExit("no Phase-A result for $CATEGORY")
PY
)"

if [[ "$SPLIT_MODE" == "calendar" ]]; then
  SPLIT_ARGS=(--split-mode calendar)
  SPLIT_TOKEN="wf7085"
else
  SPLIT_ARGS=(--split-mode ratio)
  SPLIT_TOKEN="ratio"
fi

# Map k -> ablation tag
case "$BEST_K" in
  0.5) TAG="label_k05" ;;
  0.7) TAG="label_k07" ;;
  1.0) TAG="label_k10" ;;
  1.2) TAG="label_k12" ;;
  *) echo "Unsupported k=$BEST_K"; exit 1 ;;
esac

LOG="${ROOT}/outputs/metrics/${CATEGORY}_fine_w_${SPLIT_TOKEN}.log"
mkdir -p "$(dirname "$LOG")"

declare -a WINDOWS=()
for delta in -32 -16 16 32; do
  w=$((BEST_W + delta))
  if (( w >= 20 && w <= 256 )); then
    WINDOWS+=("$w")
  fi
done

echo "=== ${CATEGORY} fine window (k=${BEST_K}, base w=${BEST_W}) ===" | tee "$LOG"

for w in "${WINDOWS[@]}"; do
  summary="${ROOT}/outputs/metrics/${CATEGORY}16_"*"_w${w}_loso_${SPLIT_TOKEN}_${TAG}_summary.json"
  if compgen -G "$summary" > /dev/null; then
    echo "[skip] w=$w exists" | tee -a "$LOG"
    continue
  fi
  echo ">>> k=${BEST_K} w=$w $(date -Iseconds)" | tee -a "$LOG"
  python3 main.py --mode loso --category "$CATEGORY" --stage all --model mlp \
    --label-k "$BEST_K" --window-size "$w" --ablation-tag "$TAG" \
    "${SPLIT_ARGS[@]}" \
    2>&1 | tee -a "$LOG"
done

python3 scripts/category_kw_compare.py --category "$CATEGORY" --split-mode "$SPLIT_MODE" \
  2>&1 | tee -a "$LOG"
