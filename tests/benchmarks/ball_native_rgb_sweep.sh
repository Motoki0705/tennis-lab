#!/usr/bin/env bash
# Run the complete sweep as one exclusive training-queue job.
set -euo pipefail
rgb_manifest=$1
rgb_model=$2
rgb_report=$3
for rgb_compile in off default; do
  PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py \
    --manifest "$rgb_manifest" --model-config "$rgb_model" \
    --output "$rgb_report/compute-${rgb_compile}.json" \
    --precision bf16 --batch-size 1 --compile-mode "$rgb_compile" --warmup 4 --steps 24
done
for rgb_compile in off default; do
  PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py \
    --manifest "$rgb_manifest" --model-config "$rgb_model" \
    --output "$rgb_report/pipeline-${rgb_compile}.json" \
    --precision bf16 --batch-size 1 --compile-mode "$rgb_compile" \
    --mode pipeline --workers 8 --pin-memory --warmup 12 --steps 96
done
