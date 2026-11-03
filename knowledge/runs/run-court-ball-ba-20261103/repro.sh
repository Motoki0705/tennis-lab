#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
input_dir=$1
videos=$2
output_dir=$3
/usr/bin/time -v -o "${output_dir}-resource-usage.txt" .venv/bin/python \
  knowledge/runs/run-court-ball-ba-20261103/run.py \
  --bundle "$input_dir/fixed_input_bundle_v1" \
  --annotations "$input_dir/human_annotations/ground_truth_v1/ground_truth.json" \
  --prior "$input_dir/foundation_v2" --videos "$videos" --output "$output_dir"
