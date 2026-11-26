#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TENNIS_REPO:?set the checkout of the recorded execution HEAD}"
expected=97cabfef23bdab3e1097af32a2cdd96cf754b9dd
if [ "$(git rev-parse HEAD)" != "$expected" ]; then
    echo "Replay requires checkout $expected" >&2
    exit 1
fi
export TENNIS_REPRO_DIR="${TENNIS_REPRO_DIR:?set by the GPU training queue}"
export PYTHONPATH="$PWD:$PWD/outputs/fasthmr-setup/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Recorded submission and capture are preserved in run.json and repro.queue.sh.
# --- original training command ---
.venv/bin/python -u "$SCRIPT_DIR/meshes_for_review.py" gvhmr-archive
.venv/bin/python -u "$SCRIPT_DIR/meshes_for_review.py" pt16
