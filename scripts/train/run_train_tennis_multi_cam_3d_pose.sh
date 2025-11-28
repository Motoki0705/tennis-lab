#!/usr/bin/env bash
set -euo pipefail

# Run tennis_multi_cam_3d_pose training via the unified CLI entrypoint.
#
# Usage:
#   ./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
#   CONFIG=configs/tennis_multi_cam_3d_pose.yaml ./scripts/train/run_train_tennis_multi_cam_3d_pose.sh
#   ./scripts/train/run_train_tennis_multi_cam_3d_pose.sh --set training.trainer.max_epochs=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_multi_cam_3d_pose.yaml}"

uv run python src/cli/tennis_multi_cam_3d_pose/train.py \
  --config "${CONFIG}" \
  "$@"
