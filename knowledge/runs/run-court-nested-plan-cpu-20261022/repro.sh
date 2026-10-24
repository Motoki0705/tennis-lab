#!/usr/bin/env bash
set -euo pipefail

# Run from the tennis-lab checkout at the base commit recorded in metrics.json.
# The archived prototype is additional experiment code; src/presets are unchanged.
: "${SCENES_ROOT:?Set SCENES_ROOT to the read-only B00-B03 scene parent directory}"
: "${PROBE_OUTPUT_DIR:?Set PROBE_OUTPUT_DIR to a NEW directory outside the input root}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python "$SCRIPT_DIR/prototype.py" \
  --scenes-root "$SCENES_ROOT" --output "$PROBE_OUTPUT_DIR" --visuals
.venv/bin/python "$SCRIPT_DIR/saved_trajectory_video.py" \
  --scenes-root "$SCENES_ROOT" --output "$PROBE_OUTPUT_DIR"
