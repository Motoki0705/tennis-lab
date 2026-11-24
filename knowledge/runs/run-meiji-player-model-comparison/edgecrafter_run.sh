#!/usr/bin/env bash
# Run one model in the shared GPU queue, in fresh processes for every speed case.
set -uo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 MODEL FIXED_SPEED_INPUTS SPEED_MANIFEST_SHA256 COHORT_MANIFEST_SHA256" >&2
    exit 2
fi

model=$1
fixed=$2
speed_sha=$3
cohort_sha=$4
experiment=knowledge/runs/run-meiji-player-model-comparison
baseline=knowledge/runs/run-person-model-speed-baseline
output="outputs/player-model-comparison/$model"
reference="$output/pt32"
export PYTHONPATH=".:$baseline" OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
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
    local exit_code
    if "$@"; then
        return 0
    else
        exit_code=$?
        status=1
        printf 'Command exited %s. Arguments' "$exit_code" >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        return "$exit_code"
    fi
}

accuracy() {
    local backend=$1 destination=$2
    shift 2
    record "${python_run[@]}" "$experiment/edgecrafter.py" evaluate \
        --model "$model" --backend "$backend" --batch-size 1 \
        --manifest-sha256 "$cohort_sha" --output "$destination" "${pose[@]}" "$@" || return
    if [[ $backend != pt32 && -f $reference/metrics.json ]]; then
        record "${python_run[@]}" "$experiment/edgecrafter_parity.py" \
            --pt32 "$reference" --candidate "$destination"
    fi
}

speed() {
    local backend=$1 destination=$2 batch
    shift 2
    for batch in 1 2; do
        record "${python_run[@]}" "$experiment/edgecrafter_speed.py" \
            --model "$model" --backend "$backend" --batch-size "$batch" \
            --inputs "$fixed" --inputs-sha256 "$speed_sha" --reference "$reference" \
            --output "$destination" "${pose[@]}" "$@" || true
    done
}

for backend in pt32 pt16; do
    if [[ $model == ecpose-* && ! -f $output/$backend/metrics.json ]]; then
        accuracy "$backend" "$output/$backend" || true
    fi
    speed "$backend" "$output/$backend"
done

trt="$output/trt16-trace2"
engine="$trt/engine"
receipts=()
if [[ ! -f $engine/export.json ]]; then
    record "${python_run[@]}" "$experiment/edgecrafter_trt.py" export \
        --model "$model" --trace-batch 2 --engine "$engine" --receipt "$trt/onnx-export.json" || true
fi
if [[ -f $trt/onnx-export.json ]]; then receipts+=(--conversion "$trt/onnx-export.json"); fi
if [[ -f $engine/export.json && ! -f $engine/engine.json ]]; then
    record "${python_run[@]}" "$experiment/edgecrafter_trt.py" build \
        --opt-batch 1 --max-batch 2 --engine "$engine" --receipt "$trt/engine-build.json" || true
fi
if [[ -f $trt/engine-build.json ]]; then receipts+=(--conversion "$trt/engine-build.json"); fi
speed trt16 "$trt" --engine "$engine" "${receipts[@]}"
if [[ -f $trt/speed-b1.json ]] && jq -e '.status == "ok"' "$trt/speed-b1.json" > /dev/null; then
    accuracy trt16 "$trt" --engine "$engine" || true
fi
exit "$status"
