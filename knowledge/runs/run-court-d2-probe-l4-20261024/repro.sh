#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'usage: bash knowledge/runs/run-court-d2-probe-l4-20261024/repro.sh RECOVERED_FAILURE_OUTPUT RECOVERED_RENDER_OPERATION' >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURT_REPRO_DIR=$(mktemp -d -p tmp court-d2-l4-repro.XXXXXX)
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python "$SCRIPT_DIR/summarize_failure.py" --recovered "$1" --operation "$2" --output "$COURT_REPRO_DIR/evidence"
cmp "$COURT_REPRO_DIR/evidence/failure-summary.json" "$SCRIPT_DIR/evidence/failure-summary.json"
cmp "$COURT_REPRO_DIR/evidence/setup-only-costs.json" "$SCRIPT_DIR/evidence/setup-only-costs.json"
.venv/bin/python "$SCRIPT_DIR/verify_probe_paths.py" --scratch "$COURT_REPRO_DIR/path-contract" --output "$COURT_REPRO_DIR/path-contract.json"
cmp "$COURT_REPRO_DIR/path-contract.json" "$SCRIPT_DIR/evidence/path-contract.json"
.venv/bin/python "$SCRIPT_DIR/verify_costs.py" --scratch "$COURT_REPRO_DIR/cost-fixture"
echo "Offline reproduction complete: $COURT_REPRO_DIR"
