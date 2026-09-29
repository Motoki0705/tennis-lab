#!/usr/bin/env bash
set -euo pipefail
if [[ "$(git rev-parse HEAD)" != "08bfc644d8bb96ad44b278394cdd3b676a456d35" ]]; then
  echo "Use a dedicated worktree checked out at 08bfc644d8bb96ad44b278394cdd3b676a456d35" >&2
  exit 2
fi
# Run from a dedicated worktree checked out at 08bfc644d8bb96ad44b278394cdd3b676a456d35.
: "${1:?Specify a new absolute output directory}"
task_repo_root="$(git rev-parse --show-toplevel)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.memory_smoke_3d --config "$task_repo_root/src/tasks/ball_refiner/refiner_3d/memory_smoke.yaml" --fixture "$task_repo_root/src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json" --output "$1" --device cpu
