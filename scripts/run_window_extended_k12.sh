#!/usr/bin/env bash
# Extended window scan above w=96 at label_k=1.2 (MLP only).
# Motivation: w=96 was the scan upper bound, not proven optimum.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/outputs/metrics/window_extended_k12.log"
mkdir -p "$(dirname "$LOG")"

# 15m bars: 128=32h, 160=40h, 192=48h
WINDOWS=(128 160 192)

echo "=== extended window scan (label_k=1.2, MLP) started $(date -Iseconds) ===" | tee "$LOG"

for w in "${WINDOWS[@]}"; do
  summary="${ROOT}/outputs/metrics/meme8_"*"_w${w}_loso_label_k12_summary.json"
  if compgen -G "$summary" > /dev/null; then
    echo "[skip] window_size=$w summary exists" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> k=1.2 window_size=$w lookback=$((w * 15 / 60))h $(date -Iseconds)" | tee -a "$LOG"
  python3 main.py --mode loso --stage all --model mlp \
    --label-k 1.2 --window-size "$w" --ablation-tag label_k12 \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/window_extended_compare.py 2>&1 | tee -a "$LOG"
echo "=== extended window scan finished $(date -Iseconds) ===" | tee -a "$LOG"
