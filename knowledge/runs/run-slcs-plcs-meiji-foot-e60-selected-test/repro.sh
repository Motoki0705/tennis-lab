#!/usr/bin/env bash
# #931: the original command read these files from outputs/; they are saved in this bundle.
set -euo pipefail
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python "$SCRIPT_DIR/evaluate_selected.py"
