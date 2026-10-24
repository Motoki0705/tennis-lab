#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'usage: bash knowledge/runs/run-court-d2-support-cpu-20261024/repro.sh SCENES_ROOT REAL_ROOT PILOT_DATASET_VERSION' >&2
  exit 2
fi
COURT_SCENES_ROOT=$1
COURT_REAL_ROOT=$2
COURT_PILOT_VERSION=$3
COURT_BUNDLE=knowledge/runs/run-court-d2-support-cpu-20261024
COURT_REPRO_DIR=$(mktemp -d -p tmp court-d2-repro.XXXXXX)
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python "$COURT_BUNDLE/diagnose.py" \
  --scenes-root "$COURT_SCENES_ROOT" --real-root "$COURT_REAL_ROOT" \
  --registry knowledge/runs/run-court-data-scaling-smoke-s42/real-evaluation-registry.json \
  --output "$COURT_REPRO_DIR/analysis"
.venv/bin/python "$COURT_BUNDLE/audit_saved.py" \
  --version "$COURT_PILOT_VERSION" --diagnosis "$COURT_REPRO_DIR/analysis" \
  --output "$COURT_REPRO_DIR/audit"
.venv/bin/python "$COURT_BUNDLE/plan_variants.py" \
  --scenes-root "$COURT_SCENES_ROOT" --output "$COURT_REPRO_DIR/plans"
.venv/bin/python "$COURT_BUNDLE/select_probe.py" \
  --plans "$COURT_REPRO_DIR/plans" --output "$COURT_REPRO_DIR/render-selection.json"
cmp "$COURT_REPRO_DIR/render-selection.json" "$COURT_BUNDLE/render-selection.json"
.venv/bin/python "$COURT_BUNDLE/costs.py" \
  --plans "$COURT_REPRO_DIR/plans" --output "$COURT_REPRO_DIR/cost-scenarios.json"
cmp "$COURT_REPRO_DIR/cost-scenarios.json" "$COURT_BUNDLE/cost-scenarios.json"
.venv/bin/python "$COURT_BUNDLE/verify_cpu.py" \
  --scenes-root "$COURT_SCENES_ROOT" --plans "$COURT_REPRO_DIR/plans" \
  --selection "$COURT_REPRO_DIR/render-selection.json" \
  --audit "$COURT_REPRO_DIR/audit/saved-pilot.json" --scratch "$COURT_REPRO_DIR/verification"
echo "CPU reproduction complete: $COURT_REPRO_DIR"
