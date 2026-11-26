#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TENNIS_REPO:?set the checkout of the recorded GPU execution HEAD}"
expected=8a25fe12b543656cb1e883e7251bd1a3fd299fd0
if [ "$(git rev-parse HEAD)" != "$expected" ]; then
    echo "FastHMR timing requires checkout $expected" >&2
    exit 1
fi
input="${PERSON_SPEED_INPUT_DIR:?set the #2497 pinned input directory}"
repro="${TENNIS_REPRO_DIR:?set by the GPU training queue}"
export PYTHONPATH="$PWD:$PWD/knowledge/runs/run-person-model-speed-baseline:$PWD/outputs/fasthmr-setup/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$repro/predictions"

# Original submission and capture are preserved in run.json and repro.queue.sh.
# --- original training command ---
status=0
for batch in 1 2 4; do
    if ! .venv/bin/python -B -u "$SCRIPT_DIR/benchmark_fasthmr.py" --input-dir "$input" --batch "$batch" --output "$repro/predictions/pt16-b$batch.json"; then
        status=1
    fi
done
exit "$status"
