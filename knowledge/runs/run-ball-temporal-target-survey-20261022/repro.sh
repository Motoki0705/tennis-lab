#!/usr/bin/env bash
set -euo pipefail

# Run from the active checkout. The source snapshot is read-only.
: "${SOURCE_REPO_ROOT:?Set SOURCE_REPO_ROOT to the repo containing the pinned outputs}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SURVEY_OUTPUT="${SURVEY_OUTPUT:-$PWD/tmp/ball-temporal-target-survey-repro}"
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python "$SCRIPT_DIR/temporal_target_probe.py" \
  --repo-root "$SOURCE_REPO_ROOT" \
  --source-probe "$SCRIPT_DIR/../run-ball-colab-transfer-feasibility-20261022/temporal-support-probe.json" \
  --output "$SURVEY_OUTPUT"
.venv/bin/python "$SCRIPT_DIR/plot_coverage.py" \
  --summary "$SURVEY_OUTPUT/summary.json" --output "$SURVEY_OUTPUT/coverage.svg"
