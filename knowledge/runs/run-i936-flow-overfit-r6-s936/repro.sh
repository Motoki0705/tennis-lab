#!/usr/bin/env bash
set -euo pipefail
TASK_DATASET="${1:?absolute verified reconditioned 12-rally dataset}"
TASK_OUTPUT="${2:?absolute new output directory}"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD" .venv/bin/python -m src.tasks.ball_refiner.scripts.training_smoke_3d --dataset "$TASK_DATASET" --config "$PWD/src/tasks/ball_refiner/refiner_3d/training_smoke.yaml" --output "$TASK_OUTPUT"
