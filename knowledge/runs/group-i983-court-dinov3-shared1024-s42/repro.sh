#!/usr/bin/env bash
set -euo pipefail
# Run from a checkout containing all four recorded evaluation bundles.
# Check the comparison implementation without changing the caller's checkout.
git diff --exit-code 75e30a565b1f7a5f7fd4f08cea2daf458d377653 -- src/tasks/court_detection/ablation/comparison.py src/tasks/court_detection/scripts/compare_ablation.py
.venv/bin/python -m src.tasks.court_detection.scripts.compare_ablation \
  --evaluations \
  knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-b-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-l-s42/evaluation/evaluation.json \
  --output "${1:?Provide a fresh comparison output directory}"
