#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
output_dir="${1:?Pass a new absolute output directory}"
case "$output_dir" in /*) ;; *) echo "Output must be absolute" >&2; exit 2;; esac
git -C "$task_repo_root" diff --quiet d0dacaf5 -- src/utils/geometry/probabilistic_triangulation src/tasks/ball_refiner/refiner_3d/synthetic src/tasks/ball_refiner/refiner_3d/triangulation.py src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml src/tasks/ball_refiner/refiner_2d/distribution.py src/tasks/blcs/generate_dataset/simulation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
"$task_repo_root/.venv/bin/python" -m src.tasks.ball_refiner.scripts.generate_synthetic_3d --project-root "$task_repo_root" --data-root /home/kamimura/projects/tennis-lab/data --plan "$task_repo_root/src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml" --output "$output_dir" --mode smoke
