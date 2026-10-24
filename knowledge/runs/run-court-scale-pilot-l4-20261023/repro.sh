#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$REPO_ROOT"
mkdir -p tmp
PILOT_CHECK_DIR=$(mktemp -d "$REPO_ROOT/tmp/court-scale-report.XXXXXX")
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python "$SCRIPT_DIR/verify_reporting.py" --output "$PILOT_CHECK_DIR/reporting"
.venv/bin/python "$SCRIPT_DIR/analyze_costs.py"
.venv/bin/python "$SCRIPT_DIR/verify_evidence.py"
