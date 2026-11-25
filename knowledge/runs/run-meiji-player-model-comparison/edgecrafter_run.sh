#!/usr/bin/env bash
# Score one model on the full cohort in the shared GPU queue.
set -uo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 MODEL COHORT_MANIFEST_SHA256" >&2
    exit 2
fi

model=$1
cohort_sha=$2
experiment=knowledge/runs/run-meiji-player-model-comparison
output="outputs/player-model-comparison/$model"
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_run=(.venv/bin/python -B -u)
pose=()
status=0

case "$model" in
    ecpose-l-o365|ecpose-x-o365) pose=(--pose-box-margin 0.10) ;;
    ecdet-l-coco|ecdet-x-coco|ecdet-l-o365|ecdet-x-o365) ;;
    *) echo "Unexpected EdgeCrafter model $model" >&2; exit 2 ;;
esac

mkdir -p "$output" || exit
if [[ -n ${TENNIS_REPRO_DIR:-} ]]; then
    realpath "$output" > "$TENNIS_REPRO_DIR/output_dir.txt" || exit
fi

record() {
    local backend=$1 phase=$2
    shift 2
    local exit_code
    if "$@"; then
        return 0
    else
        exit_code=$?
        status=1
        printf '%s %s infeasible. Phase %s exited %s. See the preceding output.\n' \
            "$model" "$backend" "$phase" "$exit_code" >&2
        return "$exit_code"
    fi
}

require_output() {
    local backend=$1 phase=$2
    shift 2
    local path
    for path in "$@"; do
        if [[ ! -s $path ]]; then
            status=1
            printf '%s %s infeasible. Phase %s did not produce %s.\n' \
                "$model" "$backend" "$phase" "$path" >&2
            return 1
        fi
    done
}

accuracy() {
    local backend=$1 destination=$2
    shift 2
    record "$backend" accuracy "${python_run[@]}" "$experiment/edgecrafter.py" evaluate \
        --model "$model" --backend "$backend" --batch-size 1 --parity not_measured \
        --manifest-sha256 "$cohort_sha" --output "$destination" "${pose[@]}" "$@" || return
    require_output "$backend" accuracy "$destination/metrics.json" "$destination/predictions.jsonl"
}

trt_accuracy() {
    local trt="$output/trt16-trace2"
    local engine="$trt/engine"
    if [[ ! -s $engine/export.json ]]; then
        record trt16 onnx_export "${python_run[@]}" "$experiment/edgecrafter_trt.py" export \
            --model "$model" --trace-batch 2 --engine "$engine" --receipt "$trt/onnx-export.json" || return
    fi
    require_output trt16 onnx_export "$engine/export.json" "$trt/onnx-export.json" || return
    if [[ ! -s $engine/engine.json ]]; then
        record trt16 engine_build "${python_run[@]}" "$experiment/edgecrafter_trt.py" build \
            --opt-batch 1 --max-batch 2 --engine "$engine" --receipt "$trt/engine-build.json" || return
    fi
    require_output trt16 engine_build "$engine/engine.json" "$trt/engine-build.json" || return
    accuracy trt16 "$trt" --engine "$engine"
}

if [[ $model == ecpose-* ]]; then
    accuracy pt16 "$output/pt16" || true
fi

trt_accuracy || true
exit "$status"
