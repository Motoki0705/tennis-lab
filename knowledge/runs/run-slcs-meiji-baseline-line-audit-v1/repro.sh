#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python knowledge/runs/run-slcs-meiji-baseline-line-audit-v1/probe.py \
  --project-root /home/kamimura/projects/tennis-lab \
  --output "${1:-/home/kamimura/projects/tennis-lab/outputs/tennis_scene/analyze/meiji_baseline_line_audit/s42-001}"
