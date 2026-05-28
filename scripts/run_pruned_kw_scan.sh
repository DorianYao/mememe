#!/usr/bin/env bash
# advice2.md Phase 1：k×window 定向精扫（仅 MLP，跳过已有 summary）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/outputs/metrics/pruned_kw_scan.log"
mkdir -p "$(dirname "$LOG")"

# k:window:tag
COMBOS=(
  "0.5:40:label_k05"
  "0.5:80:label_k05"
  "0.7:40:label_k07"
  "0.7:80:label_k07"
  "1.0:40:label_k10"
  "1.0:80:label_k10"
  "1.0:96:label_k10"
  "1.2:96:label_k12"
)

echo "=== pruned k×window scan (MLP only) started $(date -Iseconds) ===" | tee -a "$LOG"

for combo in "${COMBOS[@]}"; do
  IFS=: read -r k w tag <<< "$combo"
  summary="${ROOT}/outputs/metrics/meme8_"*"_w${w}_loso_${tag}_summary.json"
  if compgen -G "$summary" > /dev/null; then
    echo "[skip] k=$k w=$w tag=$tag summary exists" | tee -a "$LOG"
    continue
  fi
  echo "" | tee -a "$LOG"
  echo ">>> k=$k w=$w tag=$tag $(date -Iseconds)" | tee -a "$LOG"
  python3 main.py --mode loso --stage all --model mlp \
    --label-k "$k" --window-size "$w" --ablation-tag "$tag" \
    2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo ">>> compare $(date -Iseconds)" | tee -a "$LOG"
python3 scripts/pruned_kw_compare.py 2>&1 | tee -a "$LOG"
echo "=== pruned k×window scan finished $(date -Iseconds) ===" | tee -a "$LOG"
