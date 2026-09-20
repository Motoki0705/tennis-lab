#!/usr/bin/env bash
set -euo pipefail
# Use a clean checkout of the recorded commit and apply uncommitted.patch.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m scripts.analysis.evaluate_blcs_real --checkpoint blcs/real-rgb-meiji-e60-v1.ckpt --output blcs/evaluate/meiji_finetuned/s42-001
