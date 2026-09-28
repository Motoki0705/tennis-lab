#!/usr/bin/env bash
# Fixed #935 pilot: one complete training clip per source, 561 frames total.
# Run from the active checkout, exclusively through the shared training queue.
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "usage: $0 <asset_root> <detector_cache> <new_context_cache> <new_report>" >&2
    exit 2
fi
for path in "$@"; do
    [[ "$path" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
code_root="$(git rev-parse --show-toplevel)"
asset_root="$1"
evidence="$2"
context="$3"
report="$4"
python="$code_root/.venv/bin/python"
[[ ! -e "$context" && ! -e "$report" ]] || { echo "Output paths must be new" >&2; exit 2; }
mkdir -p "$(dirname "$report")"
mkdir "$report"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# This script disables CUDA visibility during compilation and never changes a
# shared binary. Reproduction builds the dependency instead of reusing a cache.
bash "$code_root/tests/benchmarks/build_dino_extension.sh" "$asset_root" "$report/dino_extension" > "$report/build-dino.log" 2>&1
export PYTHONPATH="$report/dino_extension/lib:$code_root"
"$python" -m src.tennis_scene.scripts.run_pipeline --cfg job --resolve \
    "paths.project_root=$code_root" "paths.data_root=$asset_root/data" \
    "paths.checkpoint_root=$asset_root/ckpt" "paths.artifact_root=$asset_root/outputs" \
    "paths.output_root=$asset_root/outputs" "paths.cache_root=$asset_root/data" \
    "paths.external_asset_root=$asset_root/third_party" > "$report/scene-context.yaml"
selection=(
    --clip-id tracknet/game5/Clip14
    --clip-id meiji/video_002/clip_011/cam0
    --clip-id chat_annotation/-6UwVW0DeO4__f056b9d6649bee3a__f000000496-000000821
)
generate=(
    "$python" -m src.tasks.ball_refiner.scripts.generate_context
    --store "$asset_root/data/ball_detection/ball-mix-v1" --evidence "$evidence"
    --output "$context" --scene-config "$report/scene-context.yaml" --max-tracks 64
    "${selection[@]}"
)
CUDA_VISIBLE_DEVICES='' "${generate[@]}" --dry-run > "$report/preflight.json"
if [[ -n "${TENNIS_REPRO_DIR:-}" ]]; then
    cp "$report/scene-context.yaml" "$report/preflight.json" "$report/dino_extension/build.json" "$TENNIS_REPRO_DIR/"
fi
"${generate[@]}"
# Use a tracked module, not a multiply quoted python -c shell string.
CUDA_VISIBLE_DEVICES='' "$python" "$code_root/tests/benchmarks/ball_refiner_context.py" \
    --store "$asset_root/data/ball_detection/ball-mix-v1" --evidence "$evidence" \
    --context "$context" --report "$report/context-verification.json" \
    --pose-threshold 0.15 "${selection[@]}"
if [[ -n "${TENNIS_REPRO_DIR:-}" ]]; then
    cp "$report/context-verification.json" "$TENNIS_REPRO_DIR/"
fi
