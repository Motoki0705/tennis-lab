#!/usr/bin/env bash
# CPU aggregation only. Raw inputs remain outside knowledge/runs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:?set TENNIS_REPO to the checkout containing this bundle}"
: "${PAIR_EVALUATIONS:?set PAIR_EVALUATIONS to the nine verified raw evaluation JSONs}"
: "${PAIR_COMPARISON_OUTPUT:?set PAIR_COMPARISON_OUTPUT to a new output directory}"
cd "$REPO"
.venv/bin/python "$SCRIPT_DIR/compare.py" \
  --evaluations "$PAIR_EVALUATIONS" --sources "$SCRIPT_DIR/sources.json" \
  --output "$PAIR_COMPARISON_OUTPUT"
.venv/bin/python "$SCRIPT_DIR/crosscheck.py" \
  --evaluations "$PAIR_EVALUATIONS" --bundle "$PAIR_COMPARISON_OUTPUT"
