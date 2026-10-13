#!/usr/bin/env bash
set -euo pipefail
# Run through the Colab skill on L4 with the input snapshot in run.json provisioned.
# Declare --runner-output outputs/<TRAIN_OUTPUT> when submitting this script.
# The exact observed command is preserved in request.json and run.json.
TRAIN_OUTPUT="${1:?Provide a fresh task/purpose/experiment/run-id under outputs}"
git checkout c4b12f923b243da6b502149aa6f3912c706045e7
.venv/bin/python -m src.tasks.court_detection.scripts.train \
  --config-name train_i983_s \
  "run.output_dir=$TRAIN_OUTPUT" \
  run.artifact_store.mode=rclone \
  run.artifact_store.remote=gdrive \
  "run.artifact_store.remote_root=tennis_lab/outputs/$TRAIN_OUTPUT" \
  run.artifact_store.sync_interval_seconds=60
