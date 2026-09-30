#!/usr/bin/env bash
set -euo pipefail
# Reproduce from this exact commit in a dedicated checkout; output must not exist.
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation
git checkout --detach cf406ab46afb332ed905ca25b24b4c0a1d3b1c88
timeout --signal=TERM --kill-after=10s 300s env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/.venv/bin/python -u -m src.tasks.ball_refiner.scripts.probe_conditioning_3d --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936 --training-output /home/kamimura/projects/tennis-lab/outputs/ball_refiner/train/h-dev/r10-s936-20k --output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/outputs/c936-r11-readout
