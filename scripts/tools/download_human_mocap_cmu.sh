#!/usr/bin/env bash
set -euo pipefail

# Placeholder helper for preparing human mocap BVH files.
#
# This script does NOT automatically download data (to avoid hard-coding
# brittle URLs or bypassing license agreements). Instead, it creates the
# expected directory layout and prints short instructions.
#
# Expected layout after you prepare data manually:
#   data/human_mocap/raw/
#       some_sequence_001.bvh
#       some_sequence_002.bvh
#       ...
#
# You can use, for example, CMU mocap converted to BVH or any other BVH
# dataset that provides human full-body motion. Once the BVH files are
# placed under data/human_mocap/raw, run:
#
#   ./scripts/tools/run_human_mocap_pipeline.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

RAW_DIR="data/human_mocap/raw"
mkdir -p "${RAW_DIR}"

cat <<EOF
[human-mocap] Prepared directory: ${RAW_DIR}

Please place your BVH motion capture files under this directory.
For example, you can use CMU motion capture data converted to BVH.
After copying the files, run:

  ./scripts/tools/run_human_mocap_pipeline.sh

EOF
