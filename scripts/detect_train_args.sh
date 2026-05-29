#!/usr/bin/env bash
# Shared runtime flags for category k×w scan scripts.
# Override via env: BATCH_SIZE, DATALOADER_WORKERS, DEVICE, LOSO_JOBS, FEATURE_WORKERS, NPZ_JOBS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TRAIN_EXTRA_ARGS=(--fast --skip-existing)

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
  eval "$(python3 "${ROOT}/scripts/detect_runtime.py" --shell)"
  batch="${BATCH_SIZE:-$BATCH_DEFAULT}"
  workers="${DATALOADER_WORKERS:-0}"
  TRAIN_EXTRA_ARGS+=(--device "$DEVICE" --batch-size "$batch" --dataloader-workers "$workers")
  TRAIN_EXTRA_ARGS+=(--feature-workers "$FEATURE_WORKERS")
  echo "[runtime] GPU ($DEVICE) ${RUNTIME_GPU} vram=${RUNTIME_VRAM_GB}GB tier=${RUNTIME_TIER} cpus=${RUNTIME_NCPU} batch=$batch loso_jobs=$LOSO_JOBS feature_workers=$FEATURE_WORKERS npz_jobs=$NPZ_JOBS"
else
  eval "$(python3 "${ROOT}/scripts/detect_runtime.py" --shell)"
  batch="${BATCH_SIZE:-1024}"
  TRAIN_EXTRA_ARGS+=(--dataloader-workers "${DATALOADER_WORKERS:-0}")
  TRAIN_EXTRA_ARGS+=(--feature-workers "$FEATURE_WORKERS")
  echo "[runtime] CPU tier=${RUNTIME_TIER} cpus=${RUNTIME_NCPU} loso_jobs=$LOSO_JOBS feature_workers=$FEATURE_WORKERS npz_jobs=$NPZ_JOBS"
fi
