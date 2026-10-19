#!/usr/bin/env bash
# Copy the standalone queue runtime. Also used by the deployment regression test.

set -euo pipefail

if [ "$#" -ne 1 ] || [ -z "$1" ]; then
  echo "usage: $0 DESTINATION_BIN" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUEUE_SCRIPTS="$REPOSITORY_ROOT/.agents/skills/training-queue/scripts"
DESTINATION_BIN="$1"

install -d -m 0755 "$DESTINATION_BIN" "$DESTINATION_BIN/src/utils"
install -m 0755 \
  "$QUEUE_SCRIPTS/training_queue.sh" \
  "$QUEUE_SCRIPTS/queue_history.py" \
  "$QUEUE_SCRIPTS/prune_ckpts.py" \
  "$DESTINATION_BIN/"
install -m 0644 "$REPOSITORY_ROOT/src/__init__.py" "$DESTINATION_BIN/src/"
install -m 0644 \
  "$REPOSITORY_ROOT/src/utils/__init__.py" \
  "$REPOSITORY_ROOT/src/utils/training_queue.py" \
  "$DESTINATION_BIN/src/utils/"
