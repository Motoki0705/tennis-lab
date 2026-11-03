#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 ]]; then
  echo "Usage: $0 BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON CHECKPOINT_ROOT NEW_OUTPUT_DIR" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python "$SCRIPT_DIR/run.py" compare \
  --bundle "$1" \
  --bundle-sha256 7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a \
  --videos "$2" \
  --annotations "$3" \
  --annotations-sha256 dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09 \
  --checkpoints "$4" \
  --output "$5" --video-step 1 --width 1200
