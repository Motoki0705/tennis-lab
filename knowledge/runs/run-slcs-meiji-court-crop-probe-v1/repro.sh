#!/usr/bin/env bash
set -euo pipefail
: "${TENNIS_RUN_ID:?Run through the shared training queue}"
: "${TENNIS_GPU_RESOURCE:?Shared GPU reservation required}"
export TRAINING_QUEUE_DIR=/home/kamimura/projects/tennis-lab/.training_queue
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python knowledge/runs/run-slcs-meiji-court-crop-probe-v1/probe.py "$@"
