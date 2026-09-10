#!/usr/bin/env bash
set -euo pipefail
# Run from the matching repository revision; data/assets must be provisioned.
# Full observed configuration is config.yaml beside this script.
bash scripts/colab/run.sh run blcs_axial_reference_kp14 --gpu L4 --drive-mode mount --download-to outputs/colab_downloads --keep-on-failure "$@"
