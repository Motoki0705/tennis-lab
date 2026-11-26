#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
repo="${TENNIS_REPO:?set the checkout of the recorded GPU execution HEAD}"
cd "$repo"
expected=ede444fd7a1b0b33d4624bc2be86109a1bcba4e2
if [ "$(git rev-parse HEAD)" != "$expected" ]; then
    echo "FastHMR smoke requires checkout $expected" >&2
    exit 1
fi
export TENNIS_REPRO_DIR="${TENNIS_REPRO_DIR:?set by the GPU training queue}"
export PYTHONPATH="$PWD:$PWD/outputs/fasthmr-setup/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Original queue command and checkout snapshot are in run.json and repro.queue.sh.
# --- original training command ---
status=0
for backend in pt32 pt16; do
    if ! .venv/bin/python -u "$SCRIPT_DIR/smoke_fixed_body_inputs.py" "$backend"; then
        status=1
    fi
done
exit "$status"
