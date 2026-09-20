#!/usr/bin/env bash
set -euo pipefail
# Checkout recorded commit or invoke executed_probe.py; choose a fresh output ID.
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -B knowledge/runs/run-slcs-ball-gradient-probe-v1/probe.py --output-dir outputs/slcs/analyze/ball_gradient_probe/REPRO_NEW_RUN_ID
