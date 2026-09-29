#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
export PYTHONPATH="$task_repo_root"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Use implementation commit d0dacaf5 plus this recorded bundle in a separate worktree.
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" reproduced-final.json
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" reproduced-legacy.json A0
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" reproduced-refined.json B_hi H_hi
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/diagnose_legacy.py"
