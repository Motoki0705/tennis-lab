#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
TOPK_TEST_RUN=knowledge/runs/run-ball3d-topk-multihypothesis-20261103
TOPK_TEST_TARGETS=(
  tests/unit/tennis_scene/ball_association
  tests/unit/tennis_scene/evaluation/ball3d
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py
  tests/unit/tennis_scene/visualization/test_ball3d_tables.py
  tests/unit/tennis_scene/test_configuration.py
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py
  tests/unit/tennis_scene/pipeline/test_orchestrator.py
  tests/integration/tennis_scene/test_ball3d_bundle.py
  tests/integration/tennis_scene/test_ball3d_media.py
  tests/integration/tennis_scene/test_ball3d_topk.py
)
.venv/bin/python "$TOPK_TEST_RUN/measure_tests.py" --output "$TOPK_TEST_RUN/test-collection.json" -- \
  -n 0 --collect-only -q "${TOPK_TEST_TARGETS[@]}" > "$TOPK_TEST_RUN/test-collection.txt"
.venv/bin/python "$TOPK_TEST_RUN/measure_tests.py" --output "$TOPK_TEST_RUN/test-results.json" -- \
  -n 0 --durations=0 --durations-min=0 "${TOPK_TEST_TARGETS[@]}" > "$TOPK_TEST_RUN/test-durations.txt"
tail -3 "$TOPK_TEST_RUN/test-durations.txt"
