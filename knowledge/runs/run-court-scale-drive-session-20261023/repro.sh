#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: repro.sh CHECKOUT_TMP_OUTPUT NEW_DRIVE_PATH NEW_BASELINE_DRIVE_PATH" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
.venv/bin/python "$SCRIPT_DIR/measure_transport.py" \
  --output "$1" --drive-path "$2" --baseline-drive-path "$3"
