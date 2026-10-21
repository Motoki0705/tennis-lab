#!/usr/bin/env bash
set -euo pipefail
# Run in a dedicated worktree; data/assets must be provisioned.
# Full observed configuration is config.yaml beside this script.
# Exact Colab source revision recorded in run.json and checkpoint_selection.json.
git checkout 465428b659ecafb79ce993c161cdf6d31d6e13fa
bash scripts/colab/run.sh run blcs_axial_reference_kp14 --gpu L4 --drive-mode mount --download-to outputs/colab_downloads --keep-on-failure "$@"
