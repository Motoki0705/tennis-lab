#!/usr/bin/env bash
set -euo pipefail
# Provide a NEW absolute output directory. Never overwrite the original failed run.
: "${1:?new output directory required}"
code_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$code_root"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/court_side_confidence.py" \
 --dataset /home/kamimura/projects/tennis-lab/data/blcs/single_object_camera_view_v2 \
 --confidence "$code_root/knowledge/runs/run-i935-confidence-r29-20261001/selection" \
 --rule "$code_root/src/tasks/ball_refiner/configs/confidence/meiji_val_r29.yaml" \
 --original /home/kamimura/projects/tennis-lab/outputs/court_side/evaluate/synthetic_blcs_v2/i932-holdout-s1-400-20260927/report.json \
 --output "$1"
