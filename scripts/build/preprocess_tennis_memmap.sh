#!/usr/bin/env bash
set -euo pipefail

# Preprocess tennis JSON scenes into npz/memmap arrays.
#
# This wrapper calls src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py
# using the dataset_root/name from
# configs/datasets/tennis_multi_cam_3d_pose_sim.yaml by default.
#
# Usage:
#   ./scripts/build/preprocess_tennis_memmap.sh
#   ./scripts/build/preprocess_tennis_memmap.sh --overwrite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET_CFG="${DATASET_CFG:-configs/datasets/tennis_multi_cam_3d_pose_sim.yaml}"

DATASET_ROOT="${DATASET_ROOT:-data/tennis_autogen}"

DATASET_NAME="${DATASET_NAME:-sim_fps60_dur3p0_C4_P1-20_T10}"

if [[ -z "${DATASET_NAME}" ]]; then
  echo "ERROR: failed to read dataset name from ${DATASET_CFG}" >&2
  exit 1
fi

uv run python src/cli/tennis_multi_cam_3d_pose/preprocess_memmap.py \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  "$@"
