#!/usr/bin/env bash
set -euo pipefail
# Run from repository/worktree root after applying uncommitted.patch to execution commit.
# An existing output directory is intentionally rejected; pass a new path for reproduction.
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python knowledge/runs/run-slcs-meiji-court-visual-qc-v1/probe.py \
  --project-root /home/kamimura/projects/tennis-lab \
  --samples-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-001 \
  --output "${1:-/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_court_visual_qc/s42-001}"
