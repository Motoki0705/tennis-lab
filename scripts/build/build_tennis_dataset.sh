#!/usr/bin/env bash
set -euo pipefail

# Build tennis_multi_cam_3d_pose training dataset from simulator scenes.
#
# Thin wrapper around src/cli/tennis_multi_cam_3d_pose/build_dataset.py that
# points to the canonical config used in the specs.
#
# Usage:
#   ./scripts/build/build_tennis_dataset.sh                    # normal run
#   CONFIG_PATH=... ./scripts/build/build_tennis_dataset.sh    # override config
#   ./scripts/build/build_tennis_dataset.sh --overwrite        # pass extra args

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG_PATH:-configs/tennis/build_tennis_dataset_sim.yaml}"

uv run python src/cli/tennis_multi_cam_3d_pose/build_dataset.py \
  --config "${CONFIG_PATH}" \
  "$@"
