#!/usr/bin/env bash
set -euo pipefail
if [[ "$(git rev-parse HEAD)" != "8b8a50aa1e92c50f8138e4ec4cd271add6d8f440" ]]; then
  echo "Use a dedicated worktree checked out at 8b8a50aa1e92c50f8138e4ec4cd271add6d8f440" >&2
  exit 2
fi
# Run from a dedicated worktree checked out at 8b8a50aa1e92c50f8138e4ec4cd271add6d8f440.
: "${1:?Specify a new absolute output directory}"
task_repo_root="$(git rev-parse --show-toplevel)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.generate_synthetic_3d --project-root "$task_repo_root" --data-root /home/kamimura/projects/tennis-lab/data --plan "$task_repo_root/src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml" --output "$1" --mode smoke
