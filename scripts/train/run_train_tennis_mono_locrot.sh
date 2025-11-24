#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/tennis_mono_locrot.yaml}"

uv run python src/cli/tennis_mono_locrot/train.py \
  --config "${CONFIG}" \
  "$@"
