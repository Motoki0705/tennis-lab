#!/usr/bin/env bash
set -euo pipefail
# Run in a clean checkout of cca6e5af1be1e3917f06c56ef31c4ea7930c1eb5. Dataset is the immutable H dev set.
CPU_COMPARE_OUTPUT="${1:?Pass a new absolute output directory}"
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation
git checkout cca6e5af1be1e3917f06c56ef31c4ea7930c1eb5
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_dev_baselines_3d \
  --dataset /home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936 \
  --training-output /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i936-probabilistic-triangulation/knowledge/runs/run-i936-h-dev-flow-regression-r9-s936-20260930/output \
  --output "$CPU_COMPARE_OUTPUT"
