#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; argument 1 must be a new directory.
output="${1:?usage: bash knowledge/runs/run-court-hybrid-downstream-migration/repro.sh NEW_OUTPUT_DIR}"
root="$(git rev-parse --show-toplevel)"
bundle="${root}/knowledge/runs/run-court-hybrid-downstream-migration"
command=("${root}/.venv/bin/python" -m src.tasks.court_detection.scripts.audit_hybrid_inference
  --checkpoint "${root}/ckpt/court_detection/hybrid/court-detection-epoch=17.ckpt"
  --output-dir "${output}" --scene-root "${root}/data/synthetic_data_generation/scenes")
for input in "${bundle}"/inputs/*; do command+=(--image "${input}"); done
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 "${command[@]}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH="${root}" "${root}/.venv/bin/python"   "${bundle}/smoke_downstream.py" --project-root "${root}" --output "${output}/downstream-smoke.json"
