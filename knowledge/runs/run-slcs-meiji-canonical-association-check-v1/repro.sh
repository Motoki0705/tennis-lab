#!/usr/bin/env bash
set -euo pipefail
# From repo root at run.json commit, apply uncommitted.patch to restore the
# executed probe and its review-v4 input snapshot (both absent at that commit).
# Supply a fresh, nonexistent absolute output directory; no overwrite/retry.
: "${1:?Supply a fresh absolute output directory}"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -B knowledge/runs/run-slcs-meiji-canonical-association-check-v1/probe.py --project-root /home/kamimura/projects/tennis-lab --output-dir "$1"
