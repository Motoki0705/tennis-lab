#!/usr/bin/env bash
set -euo pipefail
run="$(dirname -- "$0")"
export PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_ALLOC_CONF=expandable_segments:True
case "${1:?specify audit, probe, budget, train, collect or plot}" in
    audit|probe|budget|train)
        exec .venv/bin/python "$run/run.py" "$@"
        ;;
    collect|plot)
        exec .venv/bin/python "$run/report.py" "$@"
        ;;
    *)
        printf 'Unknown phase: %s\n' "$1" >&2
        exit 2
        ;;
esac
