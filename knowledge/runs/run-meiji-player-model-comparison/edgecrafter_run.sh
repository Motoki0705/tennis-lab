#!/usr/bin/env bash
# Score one model on the full cohort in the shared GPU queue.
set -uo pipefail

if [[ $# -ne 2 && $# -ne 4 ]]; then
    echo "Usage: $0 MODEL COHORT_MANIFEST_SHA256 [FP32_ATTENTION_ENGINE BASE_ONNX_SHA256]" >&2
    exit 2
fi
if [[ $# -eq 4 && ! $4 =~ ^[0-9a-f]{64}$ ]]; then
    echo "Specify the complete SHA-256 of the source ONNX graph" >&2
    exit 2
fi

model=$1
cohort_sha=$2
experiment=knowledge/runs/run-meiji-player-model-comparison
output="outputs/player-model-comparison/$model"
repro_output="$output"
attention_condition=
if [[ $# -eq 4 ]]; then
    attention_condition=trt16-fp32attn
    repro_output="$output/$attention_condition"
fi
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_run=(.venv/bin/python -B -u)
pose=()
status=0

case "$model" in
    ecpose-l-o365|ecpose-x-o365) pose=(--pose-box-margin 0.10) ;;
    ecdet-l-coco|ecdet-x-coco|ecdet-l-o365|ecdet-x-o365) ;;
    *) echo "Unexpected EdgeCrafter model $model" >&2; exit 2 ;;
esac

mkdir -p "$repro_output" || exit
if [[ -n ${TENNIS_REPRO_DIR:-} ]]; then
    realpath "$repro_output" > "$TENNIS_REPRO_DIR/output_dir.txt" || exit
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
    local condition=${attention_condition:-$backend}
    record "$condition" accuracy "${python_run[@]}" "$experiment/edgecrafter.py" evaluate \
        --model "$model" --backend "$backend" --batch-size 1 --parity not_measured \
        --manifest-sha256 "$cohort_sha" --output "$destination" "${pose[@]}" "$@" || return
    require_output "$condition" accuracy "$destination/metrics.json" "$destination/predictions.jsonl"
}

build_engine() {
    local condition=$1 engine=$2 receipt=$3
    if [[ ! -s $engine/engine.json ]]; then
        record "$condition" engine_build "${python_run[@]}" "$experiment/edgecrafter_trt.py" build \
            --opt-batch 1 --max-batch 2 --engine "$engine" --receipt "$receipt" || return
        require_output "$condition" engine_build "$receipt" || return
    fi
    require_output "$condition" engine_build "$engine/engine.json"
}

trt_accuracy() {
    local trt="$output/trt16-trace2"
    local engine="$trt/engine"
    if [[ ! -s $engine/export.json ]]; then
        record trt16 onnx_export "${python_run[@]}" "$experiment/edgecrafter_trt.py" export \
            --model "$model" --trace-batch 2 --engine "$engine" --receipt "$trt/onnx-export.json" || return
    fi
    require_output trt16 onnx_export "$engine/export.json" "$trt/onnx-export.json" || return
    build_engine trt16 "$engine" "$trt/engine-build.json" || return
    require_output trt16 engine_build "$trt/engine-build.json" || return
    accuracy trt16 "$trt" --engine "$engine"
}

if [[ -n $attention_condition ]]; then
    require_output "$attention_condition" onnx_rewrite "$3/export.json" || exit 1
    build_engine "$attention_condition" "$3" "$repro_output/engine-build.json" || exit 1
    require_output "$attention_condition" engine_build "$3/engine.plan" || exit 1
    accuracy trt16 "$repro_output" --engine "$3" --attention-base-onnx-sha256 "$4"
    exit "$status"
fi

if [[ $model == ecpose-* ]]; then
    accuracy pt16 "$output/pt16" || true
fi

trt_accuracy || true
exit "$status"
