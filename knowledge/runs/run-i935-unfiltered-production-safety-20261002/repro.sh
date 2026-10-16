#!/usr/bin/env bash
set -euo pipefail
: "${1:?new absolute output directory required}"
case "$1" in /*) ;; *) echo 'Output must be absolute' >&2; exit 2;; esac
# Reproduce the recorded implementation in a new, isolated worktree.
repro_main_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
repro_checkout="$repro_main_root/.claude/worktrees/repro-i935-unfiltered-production"
test -x "$repro_main_root/.venv/bin/python"
git -C "$repro_main_root" worktree add --detach "$repro_checkout" 37c41ce42c5e5b79110b07cee97088795b246f25
ln -s "$repro_main_root/.venv" "$repro_checkout/.venv"
cd "$repro_checkout"
task_repo_root="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$task_repo_root"
.venv/bin/python tests/benchmarks/court_side_unfiltered.py \
 --dataset /home/kamimura/projects/tennis-lab/data/blcs/single_object_camera_view_v2 \
 --bank /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/calibration/i935-anchored-s42-r23-20260930/residual_bank \
 --original /home/kamimura/projects/tennis-lab/outputs/court_side/evaluate/synthetic_blcs_v2/i932-holdout-s1-400-20260927/report.json \
 --previous "$task_repo_root/knowledge/runs/run-i935-correlated-safety-r30-20261001/result/report.json" \
 --output "$1"
