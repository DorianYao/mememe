#!/usr/bin/env bash
# Run strict calendar LOSO for all five research categories (16 coins each).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/validate_binance_symbols.py

python3 main.py --mode loso --all-categories --stage all --model mlp \
  --label-k 1.2 --window-size 192 --split-mode calendar \
  --ablation-tag label_k12 "$@"

python3 scripts/category_loso_compare.py --ablation-tag label_k12 --split-mode calendar
