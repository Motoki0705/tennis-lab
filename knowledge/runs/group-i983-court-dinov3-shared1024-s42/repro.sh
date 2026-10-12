#!/usr/bin/env bash
set -euo pipefail
# Run from a checkout containing all four recorded evaluation bundles.
# Check the comparison implementation without changing the caller's checkout.
git diff --exit-code c4b12f923b243da6b502149aa6f3912c706045e7 -- src/tasks/court_detection/ablation/comparison.py src/tasks/court_detection/scripts/compare_ablation.py
.venv/bin/python -m src.tasks.court_detection.scripts.compare_ablation \
  --evaluations \
  knowledge/runs/run-i983-court-dinov3-s-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-splus-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-b-s42/evaluation/evaluation.json \
  knowledge/runs/run-i983-court-dinov3-l-s42/evaluation/evaluation.json \
  --output "${1:?Provide a fresh comparison output directory}"
