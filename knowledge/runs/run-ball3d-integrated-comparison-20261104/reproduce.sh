#!/usr/bin/env bash
set -euo pipefail
# Run from the worktree root with its .venv. The frozen plan pins all executable bytes.
# Argument 1 is a NEW output directory (the official recorded directory is immutable).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BALL3D_OUTPUT="${1:?Specify a new output directory}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
.venv/bin/python "$SCRIPT_DIR/run.py" --output "$BALL3D_OUTPUT"
.venv/bin/python "$SCRIPT_DIR/report_v2.py" --output "$BALL3D_OUTPUT" \
  --videos /home/kamimura/projects/tennis-lab/data/samples
.venv/bin/python "$SCRIPT_DIR/check_cohort_sources.py" --output "$BALL3D_OUTPUT"
.venv/bin/python "$SCRIPT_DIR/finalize.py" --output "$BALL3D_OUTPUT" --portable \
  --videos /home/kamimura/projects/tennis-lab/data/samples
