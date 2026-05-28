#!/usr/bin/env bash
# label_k 扫描：LOSO × MLP/CNN，各 k 值独立 ablation_tag。
# k=0.3 baseline 已有（无 tag），此处跳过。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/outputs/metrics/label_k_scan.log"
mkdir -p "$(dirname "$LOG")"

COMMON=(
  python3 main.py
  --mode loso
  --stage all
  --model both
)

declare -A K_TO_TAG=(
  [0.1]=label_k01
  [0.2]=label_k02
  [0.5]=label_k05
  [0.7]=label_k07
  [1.0]=label_k10
)

echo "=== label_k scan started $(date -Iseconds) ===" | tee -a "$LOG"

for k in 0.1 0.2 0.5 0.7 1.0; do
  tag="${K_TO_TAG[$k]}"
  summary="${ROOT}/outputs/metrics/meme8_*_loso_${tag}_summary.json"
  if compgen -G "$summary" > /dev/null; then
    echo "[skip] k=$k tag=$tag summary already exists" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> k=$k tag=$tag $(date -Iseconds)" | tee -a "$LOG"
  "${COMMON[@]}" --label-k "$k" --ablation-tag "$tag" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/label_k_compare.py 2>&1 | tee -a "$LOG"
echo "=== label_k scan finished $(date -Iseconds) ===" | tee -a "$LOG"
