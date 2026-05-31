#!/usr/bin/env bash
# Appendix: time-decay MLP vs flat MLP on meme8 strict calendar (k=1.2, w=192).
# Single supplementary test only — no GRU or multi-category runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/scripts/detect_train_args.sh"

LOG="${ROOT}/outputs/metrics/time_decay_strict.log"
mkdir -p "$(dirname "$LOG")"

ablation="label_k12_decay"
summary="${ROOT}/outputs/metrics/meme8_"*"_w192_loso_wf7085_${ablation}_summary.json"

echo "=== time-decay strict ablation $(date -Iseconds) ===" | tee "$LOG"

if compgen -G "$summary" > /dev/null 2>/dev/null; then
  echo "[skip] mlp_decay strict summary exists" | tee -a "$LOG"
  exit 0
fi

echo ">>> model=mlp_decay ablation=$ablation $(date -Iseconds)" | tee -a "$LOG"
python3 main.py --mode loso --category meme8 --stage all --model mlp_decay \
  --label-k 1.2 --window-size 192 --split-mode calendar --ablation-tag "$ablation" \
  --skip-plots "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "=== time-decay strict finished $(date -Iseconds) ===" | tee -a "$LOG"
