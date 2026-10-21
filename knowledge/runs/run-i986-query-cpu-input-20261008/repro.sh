#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${TENNIS_REPO:?Set to a worktree at commit 8384f2fe6d6b60d0fb6487d3a23e6ee65056a1ad}"
test "$(git rev-parse HEAD)" = 8384f2fe6d6b60d0fb6487d3a23e6ee65056a1ad
: "${CPU_PROFILE_OUTPUT:?Set a new output directory}"
CPU_MANIFEST=${CPU_MANIFEST:-/home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json}
for CPU_CASE in normal preverified cached_input; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    .venv/bin/python "$SCRIPT_DIR/ball_mdd_cpu_profile.py" \
    --manifest "$CPU_MANIFEST" --output "$CPU_PROFILE_OUTPUT/$CPU_CASE.json" \
    --case "$CPU_CASE" --workers 8 --windows 96
done
