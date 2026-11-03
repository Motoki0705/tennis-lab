#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python "$SCRIPT_DIR/run.py" \
  --bundle "$1" --videos "$2" --annotations "$3" --output "$4" \
  --bundle-sha256 7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a \
  --annotations-sha256 dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09
