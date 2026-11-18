#!/usr/bin/env bash
# Restore the bundled runtime dependency, then execute the unchanged Colab record.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:?set TENNIS_REPO to a fresh tennis-lab checkout}"
task_script_dir="$REPO/outputs/tennis_scene/visualize/player_view/blender-acceptance"
mkdir -p "$task_script_dir"
(
  cd "$task_script_dir"
  if [ -e c3-preview-queue.sh ]; then
    cmp "$SCRIPT_DIR/c3-preview-queue.sh" c3-preview-queue.sh
  else
    cp "$SCRIPT_DIR/c3-preview-queue.sh" .
  fi
)
exec bash "$SCRIPT_DIR/colab-original-repro.sh"
