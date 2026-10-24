#!/usr/bin/env bash
set -euo pipefail
: "${TENNIS_RUN_ID:?Run this stage through training queue}"
: "${SCENES_ROOT:?Set the read-only input root}"
: "${PROBE_OUTPUT_DIR:?Set the prepared plan root}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
.venv/bin/python "$SCRIPT_DIR/render_probe.py" --scenes-root "$SCENES_ROOT" --plan-root "$PROBE_OUTPUT_DIR" --smoke
.venv/bin/python "$SCRIPT_DIR/render_probe.py" --scenes-root "$SCENES_ROOT" --plan-root "$PROBE_OUTPUT_DIR"
