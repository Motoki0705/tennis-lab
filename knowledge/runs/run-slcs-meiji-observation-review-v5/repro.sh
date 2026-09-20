#!/usr/bin/env bash
set -euo pipefail
# At analysis_provenance.json reproduction_commit, apply uncommitted.patch first.
# Saved videos and observation caches are required; this never runs model inference.
# Usage from the repository root: bash knowledge/runs/run-slcs-meiji-observation-review-v5/repro.sh NEW_RUN_ID
run_id=${1:?Supply a new output run ID}
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Run ID must contain only letters, digits, underscore or hyphen' >&2
  exit 2
fi
project_root=${MEIJI_PROJECT_ROOT:-/home/kamimura/projects/tennis-lab}
output_dir="outputs/tennis_scene/analyze/meiji_observation_review/$run_id"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
.venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py \
  --project-root "$project_root" \
  --observation-root "$project_root/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004" \
  --clip video_002/clip_007 --clip video_002/clip_008 --clip video_002/clip_009 \
  --output-dir "$output_dir"
.venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v5/support_probe.py \
  --project-root "$project_root" --observation-report "$output_dir/results.json" \
  --output "$output_dir/support_checks.json"
