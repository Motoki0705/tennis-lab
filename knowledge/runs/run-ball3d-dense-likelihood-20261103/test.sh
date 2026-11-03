#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
DIR="$(cd "$(dirname "$0")" && pwd)"
nodes=(
  tests/unit/tennis_scene/ball_likelihood
  tests/integration/tennis_scene/test_ball3d_dense_likelihood.py
  tests/unit/tennis_scene/evaluation/ball3d
  tests/unit/tennis_scene/test_configuration.py
  tests/unit/tennis_scene/pipeline/test_orchestrator.py
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py
  tests/unit/tennis_scene/pipeline/components/test_ball_evidence.py
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py
  tests/unit/tennis_scene/visualization/test_ball3d_tables.py
  tests/integration/tennis_scene/test_ball3d_bundle.py
  tests/integration/tennis_scene/test_ball3d_media.py
)
.venv/bin/python -m pytest -n 0 -q --collect-only "${nodes[@]}" | tee "$DIR/test-collection.txt"
.venv/bin/python -m pytest -n 0 -q --durations=0 --durations-min=0 \
  --junitxml="$DIR/test-results.xml" "${nodes[@]}" | tee "$DIR/test-durations.txt"
