#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "Usage: bash repro.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON PRIOR_BASELINE_DIR NEW_OUTPUT_DIR" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=.
.venv/bin/python knowledge/runs/run-ball3d-topk-multihypothesis-20261103/run.py \
  --bundle "$1" --videos "$2" --annotations "$3" --prior-baseline "$4" --output "$5"
