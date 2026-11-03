#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run_dir=knowledge/runs/run-court-ball-ba-20261103
nodes=(
  tests/unit/tennis_scene/ball_calibration
  tests/integration/tennis_scene/test_court_ball_ba.py
  tests/unit/tennis_scene/evaluation/ball3d
  tests/integration/tennis_scene/test_ball3d_bundle.py
  tests/unit/tennis_scene/pipeline/test_orchestrator.py
  tests/unit/tennis_scene/test_configuration.py
  tests/unit/tennis_scene/test_pipeline_configuration.py
  tests/unit/tennis_scene/pipeline/test_auto_pipeline.py
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py
  tests/unit/tennis_scene/visualization/test_ball3d_tables.py
  tests/integration/tennis_scene/test_ball3d_media.py
  tests/unit/utils/configuration/test_contracts.py::test_adapter_fields_have_truthful_default_precedence_and_absence_policy
)
.venv/bin/python -m pytest -o addopts='' --import-mode=importlib -n 0 --collect-only -q "${nodes[@]}" | tee "$run_dir/test-collection.txt"
.venv/bin/python -m pytest -o addopts='' --import-mode=importlib -n 0 -q --durations=0 --durations-min=0 \
  --junitxml=/tmp/court-ball-ba-tests.xml "${nodes[@]}" | tee "$run_dir/test-durations.txt"
.venv/bin/python "$run_dir/measure_tests.py"
