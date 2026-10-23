#!/usr/bin/env bash
set -euo pipefail

# Run from the active checkout root; choose a new directory under its tmp/.
COURT_COST_REPRO_ROOT=${1:?Specify a new local output directory under tmp/}
test ! -e "$COURT_COST_REPRO_ROOT"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RCLONE_CONFIG="$HOME/.config/tennis-lab/rclone-drive.conf"
.venv/bin/python "$SCRIPT_DIR/analyze_records.py" --output "$COURT_COST_REPRO_ROOT/analysis.json"
.venv/bin/python "$SCRIPT_DIR/measure_drive_reads.py" --output "$COURT_COST_REPRO_ROOT/read-probe"
