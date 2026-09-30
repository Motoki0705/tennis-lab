#!/usr/bin/env bash
set -euo pipefail
# Run from a checkout with the numerical file hashes in provenance.json.
# Invoke once per batch 0..9; each command is bounded independently.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_OUTPUT="${1:?absolute new audit output directory}"
TASK_BATCH="${2:?batch index 0..9}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
if (( TASK_BATCH < 3 )); then
  .venv/bin/python "$SCRIPT_DIR/audit_smoke.py" --source /home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r3 --plan "$PWD/src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml" --output "$TASK_OUTPUT" --batch "$TASK_BATCH" --batch-size 500 --workers 4
else
  .venv/bin/python "$SCRIPT_DIR/audit_balanced.py" --source /home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-smoke-r3 --plan "$PWD/src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml" --output "$TASK_OUTPUT" --batch "$TASK_BATCH" --batch-size 500 --workers 4
fi
