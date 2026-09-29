#!/usr/bin/env bash
# Run in a checkout of a5e8dc5f (see manifest.json); use a new output directory.
set -euo pipefail
output_dir="${1:?Pass a new absolute output directory}"
case "$output_dir" in /*) ;; *) echo 'Output must be absolute' >&2; exit 2;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_triangulation \
  --fixture src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json \
  --output "$output_dir/comparison.json" --trials 128 --particles 256 --hdr-samples 1024
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_triangulation \
  --fixture src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json \
  --output "$output_dir/higher-budget.json" --trials 16 --particles 1024 --hdr-samples 1024 \
  --initial-cells 24 --levels 6 --refine-cells 1024
