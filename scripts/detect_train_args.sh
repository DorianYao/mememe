#!/usr/bin/env bash
# Shared runtime flags for category k×w scan scripts.
# Override via env: BATCH_SIZE, DATALOADER_WORKERS, DEVICE

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
  batch="${BATCH_SIZE:-4096}"
  workers="${DATALOADER_WORKERS:-4}"
  TRAIN_EXTRA_ARGS+=(--device "$DEVICE" --batch-size "$batch" --dataloader-workers "$workers")
  echo "[runtime] GPU ($DEVICE) batch=$batch workers=$workers"
else
  TRAIN_EXTRA_ARGS+=(--dataloader-workers "${DATALOADER_WORKERS:-0}")
  echo "[runtime] CPU workers=${DATALOADER_WORKERS:-0}"
fi
