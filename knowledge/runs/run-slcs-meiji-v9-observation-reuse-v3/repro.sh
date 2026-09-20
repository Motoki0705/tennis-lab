#!/usr/bin/env bash
set -euo pipefail
# Historical command; report output must be a NEW path for any explicitly authorized rerun.
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb
env PYTHONPATH=. CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py --source /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004 --target /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-005 --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_v9_observation_reuse/s42-003
