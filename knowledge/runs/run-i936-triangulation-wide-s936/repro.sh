#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
output_dir="${1:?Pass a new absolute output directory}"
case "$output_dir" in /*) ;; *) echo "Output must be absolute" >&2; exit 2;; esac
git -C "$task_repo_root" diff --quiet d0dacaf5 -- src/utils/geometry/probabilistic_triangulation src/tasks/ball_refiner/refiner_3d/triangulation.py src/tasks/ball_refiner/refiner_2d/distribution.py
mkdir "$output_dir"
export PYTHONPATH="$task_repo_root"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" "$output_dir/final.json"
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" "$output_dir/legacy.json" A0
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" "$output_dir/refined.json" B_hi H_hi
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/diagnose_legacy.py" "$output_dir/legacy-diagnosis.json"
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/sensitive_case.py" "$output_dir/sensitive-case.json"
