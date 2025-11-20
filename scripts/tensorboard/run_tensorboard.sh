#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard pointing at the runs/ directory (or a custom one).
#
# Usage:
#   ./scripts/tensorboard/run_tensorboard.sh
#   RUNS_DIR=other_runs ./scripts/tensorboard/run_tensorboard.sh
#   ./scripts/tensorboard/run_tensorboard.sh --port 7007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

RUNS_DIR="${RUNS_DIR:-runs}"

exec tensorboard --logdir "${RUNS_DIR}" "$@"
