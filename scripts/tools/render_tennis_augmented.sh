#!/usr/bin/env bash
set -euo pipefail

# Render augmented TennisSceneWindowDataset samples to videos for visual inspection.
#
# By default this uses configs/datasets/tennis_pose_sim.yaml to instantiate
# the dataset and renders a few train windows into outputs/tennis_augmented_viz.
#
# Usage examples:
#   ./scripts/render_tennis_augmented.sh
#   ./scripts/render_tennis_augmented.sh --num-samples 8 --split val
#

python src/datasets/tennis/tools/render_tennis_augmented.py "$@"
