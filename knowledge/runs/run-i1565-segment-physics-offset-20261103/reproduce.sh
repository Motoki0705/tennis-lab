#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo 'usage: reproduce.sh BUNDLE_DIR VIDEOS_DIR GROUND_TRUTH_JSON PRIOR_COMPARISON_DIR NEW_OUTPUT_DIR' >&2
  exit 2
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PHYSICS_RAW_OUTPUT="${5}_predictions"
bash knowledge/runs/run-i1565-segment-physics-offset-20261103/repro.sh "$1" "$2" "$3" "$4" "$PHYSICS_RAW_OUTPUT"
PHYSICS_RAW_MANIFEST=$(cat "$PHYSICS_RAW_OUTPUT/artifacts.sha256")
.venv/bin/python knowledge/runs/run-i1565-segment-physics-offset-20261103/finalize.py \
  --bundle "$1" --annotations "$3" --prior "$PHYSICS_RAW_OUTPUT" --output "$5" \
  --prior-manifest-sha256 "$PHYSICS_RAW_MANIFEST" \
  --bundle-sha256 7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a \
  --annotations-sha256 dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09
