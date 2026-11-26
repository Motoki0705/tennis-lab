#!/usr/bin/env bash
set -euo pipefail

main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
shared=${MEIJI_POSE_SHARED_OUTPUTS:-"$main/.worktree-outputs"}
archives=${MEIJI_POSE_ARCHIVED_OUTPUTS:-"/mnt/wsl/tennis-data$main/outputs"}
workers=${MEIJI_POSE_CPU_WORKERS:-4}
run="knowledge/runs/run-multiview-pose-meiji-20261124"
mode=${1:-all}
output=${2:-outputs/i2444/acceptance}

if [[ $mode != all && $mode != meiji3 && $mode != meiji4 ]]; then
    printf 'Usage: bash %s/reproduce.sh [all|meiji3|meiji4] [output-directory]\n' "$run" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES='' PYTHONPATH=".:knowledge/runs/run-i2444-multiview-pose-probe"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ $mode == all || $mode == meiji3 ]]; then
    .venv/bin/python -B "$run/replay.py" meiji3 \
        --dataset data/tennis_multiview/processed/meiji_3cam/dataset \
        --inputs-root "$archives" --output "$output/meiji3"
fi
if [[ $mode == all || $mode == meiji4 ]]; then
    .venv/bin/python -B "$run/replay.py" meiji4 \
        --dataset "$shared/parent-2278-issue-2363-primary/i2363/i2363-all-v1/inputs/dataset" \
        --inputs-root "$shared" --output "$output/meiji4" --workers "$workers"
fi
