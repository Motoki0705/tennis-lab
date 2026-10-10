#!/usr/bin/env bash
# One exclusive training-queue job; every case gets a fresh CUDA process.
set -euo pipefail
ball_manifest=$1
ball_model=$2
ball_report=$3
mkdir -p "$ball_report"
if [[ "${4:-compute}" == pipeline ]]; then
  for ball_bs in 1 2 4; do
    for ball_workers in 0 2 4; do
      PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py \
        --manifest "$ball_manifest" --model-config "$ball_model" \
        --output "$ball_report/fp32-bs${ball_bs}-workers${ball_workers}.json" \
        --precision fp32 --batch-size "$ball_bs" --mode pipeline \
        --workers "$ball_workers" --pin-memory --steps "$((72 / ball_bs))"
    done
  done
  exit 0
fi
for ball_precision in fp32 bf16; do
  for ball_bs in 1 2 4 6 8 12 16; do
    ball_case="$ball_report/${ball_precision}-bs${ball_bs}.json"
    PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py \
      --manifest "$ball_manifest" --model-config "$ball_model" \
      --output "$ball_case" --precision "$ball_precision" --batch-size "$ball_bs"
    ball_status=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$ball_case")
    if [[ "$ball_status" == oom ]]; then break; fi
    if [[ "$ball_status" != ok ]]; then exit 1; fi
  done
done
