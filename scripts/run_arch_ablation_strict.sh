#!/usr/bin/env bash
# Appendix C: time-decay MLP + GRU vs flat MLP on meme8 strict calendar (k=1.2, w=192).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/scripts/detect_train_args.sh"

LOG="${ROOT}/outputs/metrics/arch_ablation_strict.log"
mkdir -p "$(dirname "$LOG")"

  COMMON=(
  --mode loso --category meme8 --stage all
  --label-k 1.2 --window-size 192
  --split-mode calendar
  --skip-plots
)

echo "=== arch ablation strict $(date -Iseconds) ===" | tee "$LOG"

for model in mlp mlp_decay gru; do
  case "$model" in
    mlp) ablation="label_k12" ;;
    mlp_decay) ablation="label_k12_decay" ;;
    gru) ablation="label_k12_gru" ;;
  esac
  summary="${ROOT}/outputs/metrics/meme8_"*"_w192_loso_wf7085_${ablation}_summary.json"
  if compgen -G "$summary" > /dev/null 2>/dev/null; then
    echo "[skip] $model strict summary exists ($ablation)" | tee -a "$LOG"
    continue
  fi
  echo ">>> model=$model ablation=$ablation $(date -Iseconds)" | tee -a "$LOG"
  python3 main.py "${COMMON[@]}" --model "$model" --ablation-tag "$ablation" \
    "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG"
done

echo "=== arch ablation finished $(date -Iseconds) ===" | tee -a "$LOG"
