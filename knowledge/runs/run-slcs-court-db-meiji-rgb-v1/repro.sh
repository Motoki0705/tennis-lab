#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD"
.venv/bin/python knowledge/runs/run-slcs-court-db-meiji-rgb-v1/probe.py --check
.venv/bin/python knowledge/runs/run-slcs-court-db-meiji-rgb-v1/probe.py --root "$PWD" --output "${1:-outputs/tennis_scene/analyze/court_db_meiji_rgb/s42-001}"
.venv/bin/python knowledge/runs/run-slcs-court-db-meiji-rgb-v1/summarize.py "${1:-outputs/tennis_scene/analyze/court_db_meiji_rgb/s42-001}"
