#!/usr/bin/env bash
set -euo pipefail

# Run evaluation for the tennis_multi_cam_3d_pose v1 model and render videos.
#
# Usage examples:
#   ./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose.sh
#   CONFIG=configs/tennis_multi_cam_3d_pose.yaml \
#     ./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose.sh --num-samples 8 --splits train test
#   RUNS_DIR=runs \
#     ./scripts/evaluate/run_eval_tennis_multi_cam_3d_pose.sh --start-index 16 --camera-index 1
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_multi_cam_3d_pose.yaml}"
RUNS_DIR="${RUNS_DIR:-runs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tennis_eval_videos}"

uv run python src/evaluate/tennis_multi_cam_3d_pose.py \
  --config "${CONFIG}" \
  --runs-dir "${RUNS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
