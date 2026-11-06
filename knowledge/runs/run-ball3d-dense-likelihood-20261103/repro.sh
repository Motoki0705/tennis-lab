#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" != 5 ]]; then
  echo "usage: repro.sh BUNDLE VIDEOS GROUND_TRUTH_JSON FOUNDATION_V2 NEW_OUTPUT" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python knowledge/runs/run-ball3d-dense-likelihood-20261103/run.py \
  --bundle "$1" --videos "$2" --annotations "$3" --baseline "$4" --output "$5"
