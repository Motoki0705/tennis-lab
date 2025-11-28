#!/usr/bin/env bash
set -euo pipefail

# Render human motion assets (VitPose-17 3D skeletons) on the tennis court
# using src.tennis.sim.tools.render_human_mocap_on_court.
#
# Example:
#   ./scripts/tools/render_human_mocap_on_court.sh \
#       --asset-root data/human_mocap/processed \
#       --output outputs/human_mocap_demo/demo.mp4
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

uv run python -m src.tennis.sim.tools.render_human_mocap_on_court "$@"
