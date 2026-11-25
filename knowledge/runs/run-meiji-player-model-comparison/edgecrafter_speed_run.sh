#!/usr/bin/env bash
# Time one model on the published inputs in a fresh process for each case.
set -uo pipefail

if [[ $# -ne 3 && $# -ne 4 && $# -ne 6 ]]; then
    echo "Usage: $0 MODEL FIXED_SPEED_INPUTS HARNESS_COMMIT [pt16|trt16|both|trt16-fp32attn ENGINE BASE_ONNX_SHA256]" >&2
    exit 2
fi

model=$1
fixed=$2
harness_commit=$3
if [[ ! $harness_commit =~ ^[0-9a-f]{40}$ ]]; then
    echo "Specify the complete benchmark harness commit SHA" >&2
    exit 2
fi

case "$model" in
    ecpose-l-o365|ecpose-x-o365) pose=(--pose-box-margin 0.10) ;;
    ecdet-l-coco|ecdet-x-coco|ecdet-l-o365|ecdet-x-o365) pose=() ;;
    *) echo "Unexpected EdgeCrafter model $model" >&2; exit 2 ;;
esac

requested=${4-both}
if [[ $# -eq 6 && $requested != trt16-fp32attn ]]; then
    echo "An explicit engine and source ONNX SHA-256 require trt16-fp32attn" >&2
    exit 2
fi
case $requested in
    pt16|trt16) backends=("$4") ;;
    both) backends=(pt16 trt16) ;;
    trt16-fp32attn)
        if [[ $# -ne 6 || ! ${6-} =~ ^[0-9a-f]{64}$ ]]; then
            echo "trt16-fp32attn requires its engine and the base ONNX SHA-256" >&2
            exit 2
        fi
        backends=(trt16)
        ;;
    *) echo "Select pt16, trt16, both or trt16-fp32attn" >&2; exit 2 ;;
esac

baseline=knowledge/runs/run-person-model-speed-baseline
experiment=knowledge/runs/run-meiji-player-model-comparison
output="outputs/player-model-comparison/$model"
speed="$output/speed-${harness_commit:0:9}-threads4-1"
trt="$output/trt16-trace2"
engine="$trt/engine"
repro_output="$speed"
if [[ $requested == trt16-fp32attn ]]; then
    trt="$output/$requested"
    speed="$trt/speed-${harness_commit:0:9}-threads4-1"
    engine=$5
    repro_output="$trt"
fi
export PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "$speed" || exit
if [[ -n ${TENNIS_REPRO_DIR:-} ]]; then
    realpath "$repro_output" > "$TENNIS_REPRO_DIR/output_dir.txt" || exit
fi

status=0
for backend in "${backends[@]}"; do
    args=()
    if [[ $backend == trt16 ]]; then
        if [[ ! -s $engine/engine.plan || ! -s $engine/engine.json ]]; then
            printf '%s %s infeasible. Engine artifacts missing at %s.\n' "$model" "${requested/both/trt16}" "$engine" >&2
            status=1
            continue
        fi
        args=(--engine "$engine")
        if [[ $requested == trt16-fp32attn ]]; then
            args+=(--attention-base-onnx-sha256 "$6")
            for receipt in "$trt/onnx-rewrite.json" "$trt/engine-build.json"; do
                if [[ -s $receipt ]]; then
                    args+=(--conversion "$receipt")
                fi
            done
        else
            args+=(--conversion "$trt/onnx-export.json" --conversion "$trt/engine-build.json")
        fi
    fi
    for batch in 1 2; do
        .venv/bin/python -B -u "$experiment/edgecrafter_speed.py" \
            --model "$model" --backend "$backend" --batch-size "$batch" \
            --parity not_measured --harness-commit "$harness_commit" \
            --inputs "$fixed" --inputs-sha256 64c74d75e5f1afe869911f1620b9c8e8a793fc4c4e4ac830f4abd97b21fb1a62 \
            --timing-protocol "$baseline/timing-protocol.json" \
            --timing-protocol-sha256 39e362cb87c7c373f8c1ad25e5e28025aab89b1467be21ca3139c2fb456136a2 \
            --output "$speed/$backend" "${pose[@]}" "${args[@]}" || status=1
    done
done
exit "$status"
