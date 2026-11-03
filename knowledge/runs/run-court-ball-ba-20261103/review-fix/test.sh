#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
base=knowledge/runs/run-court-ball-ba-20261103
run_dir=$base/review-fix
nodes=(
  tests/unit/utils/configuration
  tests/unit/tennis_scene/ball_calibration
  tests/integration/tennis_scene/test_court_ball_ba.py
  tests/unit/tennis_scene/evaluation/ball3d
  tests/integration/tennis_scene/test_ball3d_bundle.py
  tests/unit/tennis_scene/test_configuration.py
  tests/unit/tennis_scene/test_pipeline_configuration.py
  tests/unit/tennis_scene/pipeline/test_orchestrator.py
  tests/unit/tennis_scene/pipeline/test_auto_pipeline.py
  tests/unit/submodules/models/dino/test_extension.py
)
.venv/bin/python -m pytest -o addopts='' --import-mode=importlib -n 0 --collect-only -q \
  "${nodes[@]}" > "$run_dir/test-collection.txt" 2>&1
.venv/bin/python -m pytest -o addopts='' --import-mode=importlib -n 0 -q \
  --durations=0 --durations-min=0 --junitxml=/tmp/court-ball-ba-review-fix-tests.xml \
  "${nodes[@]}" > "$run_dir/test-durations.txt" 2>&1
.venv/bin/python "$base/measure_tests.py" --target "$run_dir" \
  --junit /tmp/court-ball-ba-review-fix-tests.xml
