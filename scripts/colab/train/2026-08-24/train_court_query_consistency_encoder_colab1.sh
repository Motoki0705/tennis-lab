#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the second half of the Issue #790 scaling grid.
# Usage after mounting Drive and checking out the PR branch:
#   bash scripts/colab/train/2026-08-24/train_court_query_consistency_encoder_colab1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[issue790] deprecated wrapper; forwarding to the 2026-08-25 scaling-grid entry point." >&2
exec bash "${SCRIPT_DIR}/../2026-08-25/train_court_query_scaling_grid_colab2.sh"
