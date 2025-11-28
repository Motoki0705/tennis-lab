#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper for human mocap on tennis court:
#   1. Prepare BVH files under data/human_mocap/raw
#   2. Convert BVH -> VitPose-17 NPZ assets
#   3. Render a demo video with multiple actors on the court
#
# Usage:
#   ./scripts/tools/run_human_mocap_pipeline.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

RAW_ROOT="data/human_mocap/raw"
PROCESSED_ROOT="data/human_mocap/processed"
OUTPUT_VIDEO="outputs/human_mocap_demo/demo.mp4"

# Step 0: Ensure raw directory exists and print short instructions.
./scripts/tools/download_human_mocap_cmu.sh

# Step 1: Convert BVH -> VitPose-17 NPZ assets.
uv run python -m src.tennis.sim.tools.preprocess_human_mocap \
  --raw-root "${RAW_ROOT}" \
  --out-root "${PROCESSED_ROOT}" \
  --min-frames 30 \
  --default-fps 30.0 \
  --verbose

# Step 2: Render a demo video with several actors on the court.
./scripts/tools/render_human_mocap_on_court.sh \
  --asset-root "${PROCESSED_ROOT}" \
  --output "${OUTPUT_VIDEO}"

printf "[human-mocap] Pipeline finished. Video written to %s\n" "${OUTPUT_VIDEO}"
