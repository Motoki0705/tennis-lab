#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "usage: bash repro.sh BUNDLE VIDEOS ANNOTATIONS NEW_OUTPUT" >&2
  exit 2
fi
/usr/bin/time -v -o "${4}.resources.txt" env PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python knowledge/runs/run-i1559-heatmap-decoding-20261103/run.py \
  --bundle "$1" --videos "$2" --annotations "$3" --output "$4"
