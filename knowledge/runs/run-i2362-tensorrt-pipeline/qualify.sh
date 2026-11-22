#!/usr/bin/env bash
set -uo pipefail

: "${TENNIS_RUN_ID:?Run through the shared training queue}"
tag=${1:?Usage: qualify.sh BUILD_TAG PHASE1_OUTPUT_ROOT DINO_EXTENSION}
phase1_root=${2:?Supply the directory containing the existing Phase 1 outputs}
export TENNIS_LAB_DINO_EXTENSION=${3:?Supply the Phase 1 DINO extension}
if [[ ! $tag =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    printf 'Invalid engine build tag: %s\n' "$tag" >&2
    exit 2
fi
jobs=knowledge/runs/run-i2362-tensorrt-pipeline
run_relative="tensorrt/qualification/$tag"
out="outputs/$run_relative"
mkdir -p "outputs/tensorrt/qualification"
mkdir "$out" || exit
printf 'model\tstage\texit\n' > "$out/status.tsv"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD:$PWD/knowledge/runs/run-i2297-clipreid-tensorrt:$PWD/knowledge/runs/run-i2303-vit-tensorrt:$PWD/knowledge/runs/run-i2304-dino-tensorrt:$PWD/knowledge/runs/run-i2305-gvhmr-tensorrt"
python=.venv/bin/python
failed=0
plans_ready=1
smoke_args=()

run() {
    local model=$1 stage=$2 code
    shift 2
    printf '[%s %s] %s\n' "$model" "$stage" "$*"
    if "$@" > "$out/$model-$stage.log" 2>&1; then
        code=0
    else
        code=$?
        failed=1
        tail -n 24 "$out/$model-$stage.log"
    fi
    printf '%s\t%s\t%s\n' "$model" "$stage" "$code" >> "$out/status.tsv"
    cat "$out/status.tsv"
    return "$code"
}

for model in clip vitpose hmr2 dino gvhmr; do
    export_args=()
    build_args=(--opt-batch 2 --max-batch 2)
    batch=2
    case "$model" in
        clip)
            input="$phase1_root/trt-env-protocol/trt-phase1a/inputs"
            shape=256x128
            ;;
        vitpose)
            input="$phase1_root/trt-vit-pose-hmr2/i2303/inputs/vitpose_h"
            shape=256x192
            ;;
        hmr2)
            input="$phase1_root/trt-vit-pose-hmr2/i2303/inputs/hmr2"
            shape=256x256
            ;;
        dino)
            input="$phase1_root/trt-dino/trt-dino/inputs"
            shape=750x1333
            export_args=(--image-size 750 1333)
            ;;
        gvhmr)
            input="$phase1_root/trt-gvhmr/trt-gvhmr/inputs"
            shape=l2-145-177
            batch=1
            build_args=(--frames 2 145 177)
            ;;
    esac
    relative="tensorrt/$model/$shape/$tag"
    directory="outputs/$relative"
    smoke_args+=(--engine "$model=$relative")
    if ! run "$model" export "$python" -B -m src.tennis_scene.scripts.tensorrt export "$model" "$directory" "${export_args[@]}"; then
        plans_ready=0
        printf '%s\tbuild\tskipped\n%s\tcompare\tskipped\n' "$model" "$model" >> "$out/status.tsv"
        continue
    fi
    if ! run "$model" build "$python" -B -m src.tennis_scene.scripts.tensorrt build "$directory" "${build_args[@]}"; then
        plans_ready=0
        printf '%s\tcompare\tskipped\n' "$model" >> "$out/status.tsv"
        continue
    fi
    run "$model" compare "$python" -B "$jobs/compare.py" "$model" --inputs "$input" --engine "$directory" \
        --batch-size "$batch" --output "$out/$model-metrics.json"
done

if (( plans_ready )); then
    run pipeline smoke "$python" -B "$jobs/smoke.py" --output "$run_relative/smoke" "${smoke_args[@]}"
else
    printf 'pipeline\tsmoke\tskipped\n' >> "$out/status.tsv"
fi
exit "$failed"
