#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_multi_cam_3d_pose_v3.yaml}"

uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config "${CONFIG}" \
  "$@"
