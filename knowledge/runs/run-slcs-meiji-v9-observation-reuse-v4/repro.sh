#!/usr/bin/env bash
set -euo pipefail
# Explicitly supply a new report directory. Execute only after src/config stabilize.
report_output=${1:?Usage: repro.sh NEW_REPORT_OUTPUT_DIRECTORY}
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb
env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python knowledge/runs/run-slcs-meiji-v9-observation-reuse-v4/reuse.py \
  --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004 \
  --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005 \
  --output "$report_output"
