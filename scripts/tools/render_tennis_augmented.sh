#!/usr/bin/env bash
set -euo pipefail

# Render augmented TennisSceneWindowDataset samples to videos for visual inspection.
#
# By default this uses configs/datasets/tennis_multi_cam_3d_pose_sim.yaml to
# instantiate the dataset and renders a few train windows into
# outputs/tennis_augmented_viz.
#
# Usage examples:
#   ./scripts/tools/render_tennis_augmented.sh
#   ./scripts/tools/render_tennis_augmented.sh --num-samples 8 --split val
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

uv run python src/datasets/tennis/tools/render_tennis_augmented.py "$@"
