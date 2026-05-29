#!/usr/bin/env bash
# Pull training metrics from remote pod to local repo (run on your WSL machine).
#
# Usage:
#   export REMOTE_HOST="root@your-pod-host"
#   export REMOTE_SSH="ssh -p 2222"          # optional
#   export REMOTE_DIR=/workspace/mememe
#   bash scripts/sync_results_from_remote.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_SSH="${REMOTE_SSH:-ssh}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/mememe}"

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Set REMOTE_HOST, e.g.:"
  echo '  export REMOTE_HOST="root@69.30.85.115"'
  echo '  export REMOTE_SSH="ssh -p 22126 -i ~/.ssh/id_ed25519"'
  echo "  bash scripts/sync_results_from_remote.sh"
  exit 1
fi

mkdir -p "$ROOT/outputs/metrics"

rsync -avz --progress -e "$REMOTE_SSH" \
  "${REMOTE_HOST}:${REMOTE_DIR}/outputs/metrics/" \
  "$ROOT/outputs/metrics/"

echo ""
echo "Synced to local: $ROOT/outputs/metrics/"
