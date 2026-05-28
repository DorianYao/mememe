#!/usr/bin/env bash
# Poll until all category OHLCV is downloaded, then launch k×w scans.
#
# Usage:
#   bash scripts/wait_and_run_category_kw.sh              # wait + Phase A (ratio)
#   bash scripts/wait_and_run_category_kw.sh calendar     # wait + strict calendar
#   POLL_SEC=120 bash scripts/wait_and_run_category_kw.sh # poll every 2 min
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPLIT_MODE="${1:-ratio}"
POLL_SEC="${POLL_SEC:-300}"

echo "Waiting for category data (poll every ${POLL_SEC}s)..."
while ! python3 scripts/check_category_data_ready.py --quiet; do
  python3 scripts/check_category_data_ready.py || true
  echo "Sleeping ${POLL_SEC}s ..."
  sleep "$POLL_SEC"
done

echo "All data ready. Starting k×w scans (split=${SPLIT_MODE})..."
bash scripts/run_all_category_kw_scans.sh "$SPLIT_MODE"
