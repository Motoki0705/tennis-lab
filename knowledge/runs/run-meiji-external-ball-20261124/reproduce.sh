#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?usage: reproduce.sh DATASET OUTPUT}
out=${2:?usage: reproduce.sh DATASET OUTPUT}
checkout=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
shared="$checkout/.worktree-outputs/court-video-shared/tennis_scene/evaluate"
bundle=knowledge/runs/run-meiji-external-ball-20261124
positions="$checkout/data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_positions"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
mkdir -p "$out"

publish() {
    local report=$1
    shift
    /usr/bin/time -v .venv/bin/python -m src.tennis_scene.scripts.import_external_ball publish \
        --dataset "$dataset" --video video_000 --positions "$positions" \
        --side-store knowledge/runs/run-court-side-video-shared-20261118/side-store \
        --side-digest a5f987bd59d40a2f20d19d7802a4435748f9e23fb61cddc0bc3a7b704d4cd668 \
        --court-store "$shared/court-video-shared-acceptance-20261117/store" \
        --drift-store "$shared/court-video-drift-lines-20261117/drift-store" \
        --report "$report" "$@"
}

publish "$out/smoke.json" --clip clip_000 --clip clip_020 2>&1 | tee "$out/smoke.log"
publish "$out/full.json" 2>&1 | tee "$out/full.log"
.venv/bin/python "$bundle/analyze.py" "$dataset" "$out/smoke.json" "$out/full.json" "$out"
.venv/bin/python "$bundle/reprojection.py" "$dataset" "$out" clip_000 clip_012 clip_020
