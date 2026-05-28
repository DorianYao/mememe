#!/usr/bin/env bash
# Label-k scan for one research category (16-coin LOSO).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CATEGORY="${1:-bluechip}"
shift || true

python3 scripts/validate_binance_symbols.py

for k_tag in "0.1:label_k01" "0.2:label_k02" "0.3:" "0.5:label_k05" "0.7:label_k07" "1.0:label_k10" "1.2:label_k12"; do
  IFS=':' read -r k ab_tag <<< "$k_tag"
  extra=()
  if [[ -n "${ab_tag}" ]]; then
    extra=(--ablation-tag "$ab_tag")
  fi
  python3 main.py --mode loso --category "$CATEGORY" --stage all --model mlp \
    --label-k "$k" --window-size 192 --split-mode calendar "${extra[@]}" "$@"
done

python3 scripts/label_k_compare.py --category "$CATEGORY"
