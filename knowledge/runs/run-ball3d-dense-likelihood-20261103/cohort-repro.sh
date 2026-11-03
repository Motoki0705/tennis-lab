#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: bash cohort-repro.sh BUNDLE GROUND_TRUTH_JSON DENSE_V2 NEW_OUTPUT' >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
exec .venv/bin/python "$SCRIPT_DIR/cohorts.py" --bundle "$1" --annotations "$2" --source "$3" --output "$4"
