#!/usr/bin/env bash
set -euo pipefail

# Run the DINOv3 patch-token-based tracking + segmentation tool on a video.
#
# By default this expects a local DINOv3 checkpoint at
#   third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
# and writes an overlay video next to the input.
#
# Usage examples:
#   ./scripts/tools/run_dinov3_patch_tracker.sh --video-path path/to/input.mp4
#   ./scripts/tools/run_dinov3_patch_tracker.sh --video-path input.mp4 --threshold 0.7
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

uv run python src/cli/tools/dinov3_patch_tracker.py "$@"
