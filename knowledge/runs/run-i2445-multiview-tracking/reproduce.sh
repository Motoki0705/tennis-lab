#!/usr/bin/env bash
set -euo pipefail

run=knowledge/runs/run-i2445-multiview-tracking
pose=knowledge/runs/run-multiview-pose-meiji-20261124
recall=knowledge/runs/run-multiview-pose-link-recall-20261124
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
archives=${MEIJI_POSE_ARCHIVED_OUTPUTS:-"/mnt/wsl/tennis-data$main/outputs"}
upstream=${MEIJI_POSE_WORKER_OUTPUTS:-"$main/.worktree-outputs/parent-1739-issue-2444-primary/i2444"}
output=${2:-outputs/i2445}
mode=${1:-all}

export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$run:$recall:$pose:knowledge/runs/run-i2444-multiview-pose-probe:."
case "$mode" in
    all|development|doubles)
        if [[ $mode != doubles ]]; then
            .venv/bin/python -B "$run/measure.py" development \
                --dataset data/tennis_multiview/processed/meiji_3cam/dataset \
                --archive "$archives" --output "$output/meiji3.json"
        fi
        if [[ $mode != development ]]; then
            .venv/bin/python -B "$run/measure.py" doubles \
                --candidate-stores "$upstream/refine/timing/meiji4" --output "$output/meiji4"
        fi
        ;;
    holdout)
        .venv/bin/python -B "$run/measure.py" holdout \
            --plan "$upstream/holdout/pose-plan.json" --poses "$upstream/holdout/vitpose" \
            --output "$output/holdout/meiji3.json"
        ;;
    *)
        printf 'Usage: bash %s/reproduce.sh [all|development|doubles|holdout] [output-directory]\n' "$run" >&2
        exit 2
        ;;
esac
