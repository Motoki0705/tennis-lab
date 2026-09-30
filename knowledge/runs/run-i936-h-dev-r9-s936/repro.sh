#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
: "${1:?absolute fresh dataset output required}"
: "${2:?absolute fresh generation-log output required}"
cd "$REPO_ROOT"
# Generation inputs were fixed at 470de1a4; this bundle's monitor is retained
# separately. The shared source calibration/geometry paths must still exist.
"$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/generate_monitor.py" "$REPO_ROOT" \
  /home/kamimura/projects/tennis-lab/data "$1" "$2"
