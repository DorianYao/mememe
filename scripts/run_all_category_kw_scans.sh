#!/usr/bin/env bash
# Run k×window tuning for all five research categories, then aggregate optimal (k, w).
#
# Phase A (default): ratio split — hyperparameter search (same protocol as meme8 tuning).
# Phase B (optional): re-run top combos under calendar split for strict validation.
#
# Usage:
#   bash scripts/run_all_category_kw_scans.sh
#   bash scripts/run_all_category_kw_scans.sh calendar
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPLIT_MODE="${1:-ratio}"
CATEGORIES=(bluechip midcap solana_fast base_eco micro_cap)

python3 scripts/check_category_data_ready.py || exit 1

echo "=== All-category k×w scan (split=${SPLIT_MODE}) ==="
for cat in "${CATEGORIES[@]}"; do
  echo ""
  echo ">>> Starting ${cat} ..."
  bash scripts/run_category_kw_scan.sh "$cat" "$SPLIT_MODE"
done

python3 scripts/category_kw_compare.py --split-mode "$SPLIT_MODE"
echo ""
echo "Done. See outputs/metrics/category_kw_optimal.json"
