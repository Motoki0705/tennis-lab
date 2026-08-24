#!/usr/bin/env bash
set -euo pipefail

# Run the first half of the Issue #790 DPT resolution/capacity grid on Colab 1.
# Usage: bash scripts/colab/train/2026-08-25/train_court_query_scaling_grid_colab1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_court_query_scaling_grid_shard.sh" colab-1
