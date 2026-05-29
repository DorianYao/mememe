#!/usr/bin/env bash
# Run one k×w LOSO combo (sequential or parallel held-outs).
# Called by category scan scripts; override LOSO_JOBS (default from detect_train_args.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOSO_JOBS="${LOSO_JOBS:-1}"
CATEGORY="$1"
K="$2"
W="$3"
TAG="$4"
shift 4

if (( LOSO_JOBS > 1 )); then
  python3 scripts/loso_parallel_train.py --jobs "$LOSO_JOBS" -- \
    python3 main.py --mode loso --category "$CATEGORY" --model mlp \
    --label-k "$K" --window-size "$W" --ablation-tag "$TAG" \
    "$@"
else
  python3 main.py --mode loso --category "$CATEGORY" --stage all --model mlp \
    --label-k "$K" --window-size "$W" --ablation-tag "$TAG" \
    "$@"
fi

# Free disk: each k×w combo leaves ~16 large .npz + .pt files (~15–25 GB).
if [[ "${AUTO_CLEANUP_NPZ:-1}" == "1" ]]; then
  split="ratio"
  prev=""
  for arg in "$@"; do
    [[ "$prev" == "--split-mode" ]] && split="$arg"
    prev="$arg"
  done
  python3 scripts/cleanup_disk.py --combo "$CATEGORY" "$K" "$W" "$TAG" "$split" \
    || true
fi
