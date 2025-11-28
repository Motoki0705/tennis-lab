#!/usr/bin/env bash
set -euo pipefail

# Render videos from v2 GT decomposition (canonical/root_trans/root_rot)
# using TennisSceneWindowDataset and the tennis_multi_cam_3d_pose visualizer.
#
# By default this uses configs/datasets/tennis_multi_cam_3d_pose_sim.yaml to
# instantiate the dataset and writes a few train windows into
# outputs/tennis_v2_decomposition_viz.
#
# Usage examples:
#   ./scripts/tools/render_tennis_v2_decomposition.sh
#   ./scripts/tools/render_tennis_v2_decomposition.sh --num-samples 8 --split val
#   ./scripts/tools/render_tennis_v2_decomposition.sh \
#       --dataset-config configs/datasets/tennis_multi_cam_3d_pose_sim.yaml \
#       --start-index 100 --num-samples 16
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

uv run python src/datasets/tennis/tools/render_tennis_v2_decomposition.py "$@"
