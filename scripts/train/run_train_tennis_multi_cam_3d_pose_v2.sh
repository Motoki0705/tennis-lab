#!/usr/bin/env bash
set -euo pipefail

# Run tennis_multi_cam_3d_pose v2 training via the v2 CLI entrypoint.
#
# Usage:
#   ./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
#   CONFIG=configs/tennis_multi_cam_3d_pose_v2.yaml ./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh
#   ./scripts/train/run_train_tennis_multi_cam_3d_pose_v2.sh --set training.trainer.max_epochs=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_multi_cam_3d_pose_v2.yaml}"

uv run python src/cli/tennis_multi_cam_3d_pose/train_v2.py \
  --config "${CONFIG}" \
  "$@"
