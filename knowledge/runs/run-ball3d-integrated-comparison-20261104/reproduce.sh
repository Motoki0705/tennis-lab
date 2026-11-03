#!/usr/bin/env bash
set -euo pipefail
# Run from the worktree root with its .venv. The frozen plan pins all executable bytes.
# Argument 1 is a NEW output directory (the official recorded directory is immutable).
BALL3D_RUN_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BALL3D_OUTPUT="${1:?Specify a new output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
.venv/bin/python "$BALL3D_RUN_DIR/run.py" --output "$BALL3D_OUTPUT"
.venv/bin/python "$BALL3D_RUN_DIR/report.py" --output "$BALL3D_OUTPUT" \
  --videos /home/kamimura/projects/tennis-lab/data/samples
