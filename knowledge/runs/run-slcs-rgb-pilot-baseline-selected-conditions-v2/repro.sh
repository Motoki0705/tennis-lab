#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python knowledge/runs/run-slcs-rgb-pilot-baseline-selected-conditions-v2/reproduce.py baseline
