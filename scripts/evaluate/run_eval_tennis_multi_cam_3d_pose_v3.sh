#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_multi_cam_3d_pose_v3.yaml}"
RUNS_DIR="${RUNS_DIR:-runs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tennis_eval_videos}"

uv run python src/evaluate/tennis_multi_cam_3d_pose.py \
  --config "${CONFIG}" \
  --runs-dir "${RUNS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
