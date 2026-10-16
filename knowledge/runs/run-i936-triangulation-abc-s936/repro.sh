#!/usr/bin/env bash
# Use the recorded implementation and a new output directory.
set -euo pipefail
output_dir="${1:?Pass a new absolute output directory}"
case "$output_dir" in /*) ;; *) echo 'Output must be absolute' >&2; exit 2;; esac
repro_main_root="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
repro_checkout="$repro_main_root/.claude/worktrees/repro-i936-triangulation-abc"
test -x "$repro_main_root/.venv/bin/python"
git -C "$repro_main_root" worktree add --detach "$repro_checkout" a5e8dc5f7bb4f23a8601ba1f3d5483f850eeda99
ln -s "$repro_main_root/.venv" "$repro_checkout/.venv"
cd "$repro_checkout"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_triangulation \
  --fixture src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json \
  --output "$output_dir/comparison.json" --trials 128 --particles 256 --hdr-samples 1024
.venv/bin/python -m src.tasks.ball_refiner.scripts.compare_triangulation \
  --fixture src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json \
  --output "$output_dir/higher-budget.json" --trials 16 --particles 1024 --hdr-samples 1024 \
  --initial-cells 24 --levels 6 --refine-cells 1024
