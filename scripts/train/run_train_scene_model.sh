#!/usr/bin/env bash
set -euo pipefail

# Run SceneModel training via the unified CLI entrypoint.
#
# Usage:
#   ./scripts/train/run_train_scene_model.sh
#   CONFIG=configs/scene_model_debug.yaml ./scripts/train/run_train_scene_model.sh
#   ./scripts/train/run_train_scene_model.sh --set training.trainer.max_epochs=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/scene_model.yaml}"

uv run python src/cli/scene_model/train.py \
  --config "${CONFIG}" \
  "$@"
