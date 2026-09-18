#!/usr/bin/env bash
set -euo pipefail
# Use a clean checkout of the recorded commit and apply uncommitted.patch.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m scripts.analysis.evaluate_blcs_real --checkpoint blcs/blcs-axial-reference-kp14-corners-v3-4-t128-seed42-best.ckpt --output blcs/evaluate/meiji_baseline/s42-001
