#!/usr/bin/env bash
# window_size 扫描（固定 label_k=1.0）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/outputs/metrics/window_scan_k1.log"
mkdir -p "$(dirname "$LOG")"

COMMON=(
  python3 main.py
  --mode loso
  --stage all
  --model both
  --label-k 1.0
  --ablation-tag label_k10
)

echo "=== window_size scan (label_k=1.0) started $(date -Iseconds) ===" | tee -a "$LOG"

for w in 20 40 60 96; do
  summary="${ROOT}/outputs/metrics/meme8_"*"_w${w}_loso_label_k10_summary.json"
  if compgen -G "$summary" > /dev/null; then
    echo "[skip] window_size=$w summary already exists" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> window_size=$w label_k=1.0 $(date -Iseconds)" | tee -a "$LOG"
  "${COMMON[@]}" --window-size "$w" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/window_scan_compare.py 2>&1 | tee -a "$LOG"
echo "=== window_size scan finished $(date -Iseconds) ===" | tee -a "$LOG"
