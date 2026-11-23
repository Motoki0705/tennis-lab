#!/usr/bin/env bash
set -euo pipefail

checkout=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
shared_outputs=${I2444_SHARED_OUTPUTS:-"$checkout/.worktree-outputs"}
archived_outputs=${I2444_ARCHIVED_OUTPUTS:-"/mnt/wsl/tennis-data$checkout/outputs"}
output=${1:-outputs/i2444/planning/replay}
run=knowledge/runs/run-i2444-multiview-pose-probe

mkdir -p "$output"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -B "$run/probe.py" \
    --dataset "$shared_outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1/inputs/dataset" \
    --artifact-root "$shared_outputs" \
    --ball-run "$shared_outputs/court-side-video-shared/tennis_scene/evaluate/court-side-20261118" \
    --stride 12 > "$output/side.json"
.venv/bin/python -B "$run/assignment_probe.py" \
    --dataset data/tennis_multiview/processed/meiji_3cam/dataset \
    --archives "$archived_outputs" > "$output/pair-f1.json"
