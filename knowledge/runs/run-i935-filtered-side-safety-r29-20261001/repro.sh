#!/usr/bin/env bash
set -euo pipefail
# Provide a NEW absolute output directory. Never overwrite the original failed run.
: "${1:?new output directory required}"
task_repo_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
cd "$task_repo_root"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$task_repo_root"
"$task_repo_root/.venv/bin/python" "$task_repo_root/tests/benchmarks/court_side_confidence.py" \
 --dataset /home/kamimura/projects/tennis-lab/data/blcs/single_object_camera_view_v2 \
 --confidence "$task_repo_root/knowledge/runs/run-i935-confidence-r29-20261001/selection" \
 --rule "$task_repo_root/src/tasks/ball_refiner/configs/confidence/meiji_val_r29.yaml" \
 --original /home/kamimura/projects/tennis-lab/outputs/court_side/evaluate/synthetic_blcs_v2/i932-holdout-s1-400-20260927/report.json \
 --output "$1"
