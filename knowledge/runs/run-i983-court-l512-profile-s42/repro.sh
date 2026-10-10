#!/usr/bin/env bash
set -euo pipefail
# Run through the Colab skill on L4 with inputs.json assets provisioned and --persist.
# Pass a fresh output directory; existing artifacts must not be overwritten.
PROFILE_OUTPUT="${1:?Provide a fresh profile output directory}"
git checkout 98b737829ba8baae8b823c7b7728cf0608f5073f
.venv/bin/python -m src.tasks.court_detection.scripts.profile_ablation --output "$PROFILE_OUTPUT" --steps 3
