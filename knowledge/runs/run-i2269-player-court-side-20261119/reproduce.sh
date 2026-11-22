#!/usr/bin/env bash
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
common=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
script="$repo/knowledge/runs/run-i2269-player-court-side-20261119/evaluate.py"
cd "$repo"
run_directory=${1:-outputs/tennis_scene/evaluate/court-side-player-20261119}
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python "$script" --common-outputs "$common/.worktree-outputs" --run-directory "$run_directory"
