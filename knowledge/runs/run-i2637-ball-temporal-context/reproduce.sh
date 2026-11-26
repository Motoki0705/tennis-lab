#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
    printf '%s\n' 'usage from repo root: reproduce.sh cpu|gpu STORE MANIFEST OUTPUT_DIR' >&2
    exit 2
fi
stage=$1
store=$2
manifest=$3
output=$4
script="$(dirname -- "$0")/evaluate.py"
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "$output"

case "$stage" in
    cpu)
        export CUDA_VISIBLE_DEVICES=""
        .venv/bin/python "$script" plan --store "$store" --manifest "$manifest" --output "$output/anchors.json"
        .venv/bin/python "$script" shape --store "$store" --plan "$output/anchors.json" --output "$output/shape.json"
        ;;
    gpu)
        if [[ -v TENNIS_REPRO_DIR ]]; then
            realpath -- "$output" > "$TENNIS_REPRO_DIR/output_dir.txt"
        fi
        .venv/bin/python "$script" gpu --store "$store" --plan "$(dirname -- "$script")/anchors.json" --output "$output/temporal.json"
        ;;
    *)
        printf 'unknown stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac
