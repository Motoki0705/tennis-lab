#!/usr/bin/env bash
# Added staging wrapper. colab-repro.sh is the exact kg_from_colab.py output.
# Run only in a separately provisioned VM, with an authenticated RCLONE_CONFIG.
# This replays the empty service request; it does not enqueue scientific work.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${TENNIS_REPO:?set TENNIS_REPO to the disposable /content/tennis-lab checkout}"
: "${RCLONE_CONFIG:?set an authenticated rclone configuration}"
python3 "$SCRIPT_DIR/prepare_control.py" \
  --destination /content/.tennis-colab/court-scene-pair-20261110-recovery5/control
bash "$SCRIPT_DIR/colab-repro.sh"
