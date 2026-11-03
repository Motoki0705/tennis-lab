#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
.venv/bin/python knowledge/runs/run-ball3d-candidate-gmm-20261103/run.py \
  --bundle "$1" --videos "$2" --annotations "$3" --prior-baseline "$4" --output "$5"
