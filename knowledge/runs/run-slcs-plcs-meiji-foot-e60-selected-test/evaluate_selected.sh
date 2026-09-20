#!/usr/bin/env bash
set -euo pipefail
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python outputs/plcs/analyze/meiji_foot_final/s42-001/evaluate_selected.py
