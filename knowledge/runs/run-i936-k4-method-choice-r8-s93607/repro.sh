#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
: "${1:?absolute fresh output root required}"
RUN_OUTPUT="$1"
case "$RUN_OUTPUT" in /*) ;; *) exit 2 ;; esac
cd "$REPO_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT"
# sample.json's nine original synthetic NPZs must still exist at their recorded
# absolute paths and hashes. No real-ball input or calibration replacement.
for method in A H ray adaptive_ray C Q20; do
    timeout 1150 "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/compare.py" --method "$method" --output "$RUN_OUTPUT/$method"
done
"$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/collect.py" --results "$RUN_OUTPUT"
