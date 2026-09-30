#!/usr/bin/env bash
set -euo pipefail
# Use a dedicated checkout. Baseline numerical source: 32cc5d02; adaptive: 1141aed3.
# SCRIPT_DIR may point to this later evidence bundle from either checkout.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_OUTPUT="${1:?absolute new output directory}"
TASK_PHASE="${2:?before or after}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
case "$TASK_PHASE" in
  before)
    .venv/bin/python "$SCRIPT_DIR/audit.py" --sample "$SCRIPT_DIR/sample.json" --output "$TASK_OUTPUT" --workers 4
    ;;
  after)
    .venv/bin/python "$SCRIPT_DIR/audit.py" --sample "$SCRIPT_DIR/sample.json" --settings "$SCRIPT_DIR/adaptive-settings.json" --output "$TASK_OUTPUT" --workers 4
    ;;
  *) exit 2 ;;
esac
