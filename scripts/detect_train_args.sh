#!/usr/bin/env bash
# Shared runtime flags for category k×w scan scripts.
# Override via env: BATCH_SIZE, DATALOADER_WORKERS, DEVICE, LOSO_JOBS, FEATURE_WORKERS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_EXTRA_ARGS=(--fast --skip-existing --skip-plots)

if [[ -n "${DEVICE:-}" ]]; then
  use_cuda=0
  [[ "$DEVICE" == cuda* ]] && use_cuda=1
elif python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  use_cuda=1
  DEVICE=cuda
else
  use_cuda=0
  DEVICE=cpu
fi

if (( use_cuda )); then
  batch="${BATCH_SIZE:-16384}"
  workers="${DATALOADER_WORKERS:-0}"
  ncpu="$(python3 "${ROOT}/scripts/effective_cpus.py" 2>/dev/null || nproc 2>/dev/null || echo 4)"
  export LOSO_JOBS="${LOSO_JOBS:-4}"
  if [[ -z "${FEATURE_WORKERS:-}" ]]; then
    feat=$(( ncpu > 1 ? ncpu - 1 : 1 ))
    (( feat > 8 )) && feat=8
    export FEATURE_WORKERS="$feat"
  fi
  TRAIN_EXTRA_ARGS+=(--device "$DEVICE" --batch-size "$batch" --dataloader-workers "$workers")
  TRAIN_EXTRA_ARGS+=(--feature-workers "$FEATURE_WORKERS")
  echo "[runtime] GPU ($DEVICE) cpus=$ncpu batch=$batch workers=$workers loso_jobs=$LOSO_JOBS feature_workers=$FEATURE_WORKERS"
else
  ncpu="$(python3 "${ROOT}/scripts/effective_cpus.py" 2>/dev/null || nproc 2>/dev/null || echo 4)"
  export LOSO_JOBS="${LOSO_JOBS:-$(( ncpu > 2 ? ncpu / 2 : 2 ))}"
  if [[ -z "${FEATURE_WORKERS:-}" ]]; then
    feat=$(( ncpu > 1 ? ncpu - 1 : 1 ))
    (( feat > 8 )) && feat=8
    export FEATURE_WORKERS="$feat"
  fi
  TRAIN_EXTRA_ARGS+=(--dataloader-workers "${DATALOADER_WORKERS:-0}")
  TRAIN_EXTRA_ARGS+=(--feature-workers "$FEATURE_WORKERS")
  echo "[runtime] CPU cpus=$ncpu loso_jobs=$LOSO_JOBS feature_workers=$FEATURE_WORKERS"
fi
