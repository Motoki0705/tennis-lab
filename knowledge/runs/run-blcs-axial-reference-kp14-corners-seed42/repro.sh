#!/usr/bin/env bash
set -euo pipefail
# Run in a dedicated worktree; data/assets must be provisioned.
# Full observed configuration is config.yaml beside this script.
# Exact Colab source revision recorded in run.json and checkpoint_selection.json.
git checkout 41da7f0aedbad30bf0cc65852b2ee979b9e05d11
bash scripts/colab/run.sh run blcs_axial_reference_kp14 --gpu L4 --drive-mode mount --download-to outputs/colab_downloads --keep-on-failure "$@"
