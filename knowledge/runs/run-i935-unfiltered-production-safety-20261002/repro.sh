#!/usr/bin/env bash
set -euo pipefail
: "${1:?new absolute output directory required}"
# Run from the checkout containing the commit recorded in repro.json.
task_repo_root="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$task_repo_root"
"$task_repo_root/.venv/bin/python" "$task_repo_root/tests/benchmarks/court_side_unfiltered.py" \
 --dataset /home/kamimura/projects/tennis-lab/data/blcs/single_object_camera_view_v2 \
 --bank /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/calibration/i935-anchored-s42-r23-20260930/residual_bank \
 --original /home/kamimura/projects/tennis-lab/outputs/court_side/evaluate/synthetic_blcs_v2/i932-holdout-s1-400-20260927/report.json \
 --previous "$task_repo_root/knowledge/runs/run-i935-correlated-safety-r30-20261001/result/report.json" \
 --output "$1"
