#!/usr/bin/env bash
# Archived original command is in colab-original-repro.sh; prepare a NEW owned VM
# and explicit fresh budget/inputs/preview first. This entry never allocates one.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:?set a separate pinned tennis-lab checkout}"
: "${TENNIS_REPRO_PREPARED_OWNED_VM:?prepare a new owned VM and fresh explicit budget first}"
cd "$REPO"
test "$(git rev-parse HEAD)" = d133993421c206af28340f4ffe082c65b70fe7a7
OPS=outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2
mkdir -p "$OPS"
for name in run_full_queue.py render_full_comparison.py archive_render_frames.py full-driver-source-digests.json; do
  if [ -e "$OPS/$name" ]; then cmp "$SCRIPT_DIR/$name" "$OPS/$name"; else cp "$SCRIPT_DIR/$name" "$OPS/$name"; fi
done
# --- original training command ---
.venv/bin/python "$SCRIPT_DIR/run_full_queue.py"
