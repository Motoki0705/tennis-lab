#!/usr/bin/env bash
set -euo pipefail

# Run Issue #790 encoder depths 1/2/4/8 with seed 43 on Colab 0.
# Usage after mounting Drive and checking out the PR branch:
#   bash scripts/colab/train/2026-08-24/train_court_query_consistency_encoder_colab0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_court_query_consistency_encoder_shard.sh" colab-0 43
