#!/usr/bin/env bash
# Run one RT-DETRv4 variant under an exclusive training-queue reservation.
set -uo pipefail

if [[ $# -ne 4 || -z ${TENNIS_RUN_ID:-} || -z ${TENNIS_REPRO_DIR:-} ]]; then
    printf 'In the training queue, run %s MODEL SPEED_INPUTS SPEED_SHA256 COHORT_SHA256\n' "$0" >&2
    exit 2
fi

model=$1
case "$model" in
    rtdetrv4-l) size=L ;;
    rtdetrv4-x) size=X ;;
    *) printf 'Unknown RT-DETRv4 model %s\n' "$model" >&2; exit 2 ;;
esac

inputs=$2
speed_sha=$3
cohort_sha=$4
parity_reason='PT32未計測'
benchmark=data/tennis_multiview/processed/meiji_4cam/benchmarks
baseline=knowledge/runs/run-person-model-speed-baseline
root=outputs/player-model-comparison
assets="$root/upstream"
results="$root/$model"
trt="$results/trt16/trace-b2"
engine="$trt/engine"
export PYTHONPATH=".:$assets/python-deps:$baseline"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "$results" || exit
realpath "$results" > "$TENNIS_REPRO_DIR/output_dir.txt" || exit
runner=(.venv/bin/python -B -u knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.py
    --model-id "$model" --upstream "$assets/RT-DETRv4" --weights "$assets/RTv4-$size-hgnet.pth")
status=0

record() {
    local code
    "$@"
    code=$?
    if ((code != 0)); then
        status=1
        printf 'Exit %s from' "$code" >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
    fi
    return "$code"
}

accuracy() {
    local backend=$1
    shift
    record "${runner[@]}" accuracy --device cuda:0 --backend "$backend" \
        --source "$benchmark/multiview-v1" --benchmark "$benchmark/multiview-player-evaluation-v1" \
        --manifest-sha256 "$cohort_sha" --output "$results/$backend" \
        --parity-not-evaluated-reason "$parity_reason" "$@"
}

speed() {
    local backend=$1 batch
    shift
    for batch in 1 2; do
        record "${runner[@]}" speed --device cuda:0 --backend "$backend" --batch-size "$batch" \
            --inputs "$inputs" --inputs-sha256 "$speed_sha" --output "$results/$backend" \
            --parity-not-evaluated-reason "$parity_reason" "$@"
    done
}

accuracy pt16
speed pt16

if [[ ! -f "$engine/export.json" ]]; then
    record "${runner[@]}" export --engine-dir "$engine" --receipt "$trt/onnx-export.json"
fi
if [[ -f "$engine/export.json" && ! -f "$engine/engine.json" ]]; then
    record "${runner[@]}" build --device cuda:0 --engine-dir "$engine" \
        --receipt "$trt/engine-build.json" --opt-batch 2 --max-batch 2 --workspace-gib 1
fi

receipts=()
for conversion in onnx-export engine-build; do
    if [[ -f "$trt/$conversion.json" ]]; then
        receipts+=(--conversion "$trt/$conversion.json")
    fi
done
if [[ -f "$engine/engine.json" ]]; then
    accuracy trt16 --engine-dir "$engine"
else
    printf 'TRT16 accuracy requires a valid engine for %s\n' "$model" >&2
    status=1
fi
speed trt16 --engine-dir "$engine" "${receipts[@]}"
exit "$status"
