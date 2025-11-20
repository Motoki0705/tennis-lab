#!/usr/bin/env bash
set -euo pipefail

# Build tennis pose training dataset from simulator scenes.
#
# This is a thin wrapper around src/cli/build_tennis_dataset.py that
# points to the canonical config used in the specs.
#
# Usage:
#   ./scripts/build_tennis_dataset.sh            # normal run
#   ./scripts/build_tennis_dataset.sh --overwrite  # allow overwrite
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-configs/tennis/build_tennis_dataset_sim.yaml}"

python src/cli/build_tennis_dataset.py \
  --config "${CONFIG_PATH}" \
  "$@"
