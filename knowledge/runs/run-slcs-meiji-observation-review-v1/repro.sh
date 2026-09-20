#!/usr/bin/env bash
set -euo pipefail
# Run from repo root at the recorded reproduction commit; use a fresh output run ID.
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. .venv/bin/python -B knowledge/runs/run-slcs-meiji-observation-review-v1/probe.py --project-root /home/kamimura/projects/tennis-lab --observation-root /home/kamimura/projects/tennis-lab/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004 --clip video_000/clip_000 --clip video_000/clip_001 --clip video_000/clip_002 --output-dir outputs/tennis_scene/analyze/meiji_observation_review/REPRO_NEW_RUN_ID
