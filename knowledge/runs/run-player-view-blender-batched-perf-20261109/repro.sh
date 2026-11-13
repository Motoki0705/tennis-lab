#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${TENNIS_REPO:?Set TENNIS_REPO to a fresh tennis-lab checkout}
task_script_dir="$REPO/outputs/tennis_scene/visualize/player_view/blender-acceptance"
mkdir -p "$task_script_dir"
(
  cd "$task_script_dir"
  for name in run_perf_batched_queue.sh resume_batched.py; do
    if [[ -e "$name" ]]; then
      cmp "$SCRIPT_DIR/$name" "$name"
    else
      cp "$SCRIPT_DIR/$name" .
    fi
  done
)
exec bash "$SCRIPT_DIR/colab-original-repro.sh"
