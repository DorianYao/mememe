#!/usr/bin/env bash
# Phase A interim (k,w) strict calendar LOSO for categories missing wf7085 summaries.
# Usage: bash scripts/run_category_strict_spotcheck.sh [category ...]
# Default: bluechip midcap solana_fast micro_cap (base_eco already validated).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=detect_train_args.sh
source "${ROOT}/scripts/detect_train_args.sh"

LOG="${ROOT}/outputs/metrics/category_strict_spotcheck.log"
mkdir -p "$(dirname "$LOG")"

# category:k:w:tag
DEFAULT_MATRIX=(
  "bluechip:1.2:128:label_k12"
  "midcap:1.2:96:label_k12"
  "solana_fast:1.2:192:label_k12"
  "micro_cap:1.2:96:label_k12"
)

if (($# > 0)); then
  REQUESTED=("$@")
  MATRIX=()
  for cat in "${REQUESTED[@]}"; do
    case "$cat" in
      bluechip) MATRIX+=("bluechip:1.2:128:label_k12") ;;
      midcap) MATRIX+=("midcap:1.2:96:label_k12") ;;
      solana_fast) MATRIX+=("solana_fast:1.2:192:label_k12") ;;
      micro_cap) MATRIX+=("micro_cap:1.2:96:label_k12") ;;
      base_eco) MATRIX+=("base_eco:1.5:224:label_k15") ;;
      *) echo "Unknown category: $cat" >&2; exit 1 ;;
    esac
  done
else
  MATRIX=("${DEFAULT_MATRIX[@]}")
fi

python3 scripts/check_category_data_ready.py --quiet || {
  echo "ERROR: run python3 scripts/download_all_category_data.py" >&2
  exit 1
}
python3 scripts/validate_binance_symbols.py

echo "=== category strict spot-check started $(date -Iseconds) ===" | tee "$LOG"

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

for entry in "${MATRIX[@]}"; do
  IFS=: read -r category k w tag <<< "$entry"
  summary="${ROOT}/outputs/metrics/${category}16_"*"_w${w}_loso_wf7085_${tag}_summary.json"
  if compgen -G "$summary" > /dev/null && _summary_complete "$summary" 16; then
    echo "[skip] $category k=$k w=$w strict summary complete" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> strict $category k=$k w=$w tag=$tag $(date -Iseconds)" | tee -a "$LOG"
  bash scripts/run_loso_combo.sh "$category" "$k" "$w" "$tag" \
    "${TRAIN_EXTRA_ARGS[@]}" \
    --split-mode calendar --stage all \
    --skip-plots \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> category_loso_compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/category_loso_compare.py --ablation-tag label_k12 --split-mode calendar \
  2>&1 | tee -a "$LOG" || true
python3 scripts/build_category_paper_matrix.py 2>&1 | tee -a "$LOG" || true
echo "=== category strict spot-check finished $(date -Iseconds) ===" | tee -a "$LOG"
