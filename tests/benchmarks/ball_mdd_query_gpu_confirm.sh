#!/usr/bin/env bash
# Same 108 sampled windows for every case: 12 warmup + 96 measured.
set -euo pipefail
ball_manifest=$1
ball_model=$2
ball_report=$3
mkdir -p "$ball_report"
for ball_setting in 1:4 2:4 4:4 1:6 2:6 1:8 2:8; do
  ball_bs=${ball_setting%:*}
  ball_workers=${ball_setting#*:}
  PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py \
    --manifest "$ball_manifest" --model-config "$ball_model" \
    --output "$ball_report/bf16-bs${ball_bs}-workers${ball_workers}.json" \
    --precision bf16 --batch-size "$ball_bs" --mode pipeline \
    --workers "$ball_workers" --pin-memory --warmup "$((12 / ball_bs))" --steps "$((96 / ball_bs))"
done
