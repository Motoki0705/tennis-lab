#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 3 ]]; then
    echo "Usage: $0 DETECTOR_CHECKOUT SINGLE_OBJECT_DATASET OUTPUT_JSON" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR_CHECKOUT="$(realpath -- "$1")"
DATASET_ROOT="$(realpath -- "$2")"
METRICS_OUTPUT="$(realpath -m -- "$3")"
DETECTOR_COMMIT=aac549c1181e8e34105a00d54fabe8f93f8ed700

cd -- "$DETECTOR_CHECKOUT"
if [[ "$(git rev-parse HEAD)" != "$DETECTOR_COMMIT" ]] || [[ -n "$(git status --porcelain -- src/)" ]]; then
    echo "Use the committed detector source at $DETECTOR_COMMIT" >&2
    exit 2
fi
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python "$SCRIPT_DIR/evaluate.py" --dataset "$DATASET_ROOT" --output "$METRICS_OUTPUT"
cmp -- "$SCRIPT_DIR/metrics.json" "$METRICS_OUTPUT"
