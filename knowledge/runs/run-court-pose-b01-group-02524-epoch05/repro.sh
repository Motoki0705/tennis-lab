#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
checkpoint="$repo_root/outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/court-detection-epoch=05.ckpt"
dataset_root="$repo_root/data/synthetic_data_generation/scenes/B01/datasets/court"
runtime_config="$repo_root/knowledge/runs/run-court-pose-b01-group-02524-epoch05/config.yaml"
output_dir="${1:-$repo_root/outputs/court-pose-b01-group-02524-epoch05-repro}"

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 "$repo_root/.venv/bin/python" \
  -m src.tasks.court_detection.scripts.evaluate_pose_trajectory \
  --checkpoint "$checkpoint" \
  --runtime-config "$runtime_config" \
  --runtime-project-root "$repo_root" \
  --dataset-root "$dataset_root" \
  --trajectory-group-id group-02524 \
  --output-dir "$output_dir"
