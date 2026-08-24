#!/usr/bin/env bash
set -euo pipefail

# Run one deterministic half of the Issue #790 DPT scaling grid.
# The canonical v3 manifest owns run IDs, architecture taps, and evidence paths;
# this wrapper only relocates runtime roots and selects one 12-run shard.

if [[ "$#" -ne 1 || "$1" != "colab-1" && "$1" != "colab-2" ]]; then
    echo "Usage: $0 {colab-1|colab-2}" >&2
    exit 2
fi

SHARD_NAME="$1"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/content/drive/MyDrive/tennis_lab/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}}"
DRIVE_QUEUE_ROOT="${DRIVE_QUEUE_ROOT:-/content/drive/MyDrive/tennis_lab/training_queue}"
TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR:-${DRIVE_QUEUE_ROOT}/issue-790/grid-v3-${SHARD_NAME}}"
SESSION_ID="${SESSION_ID:-issue790-grid-${SHARD_NAME}}"
QUEUE_SCRIPT="${REPO_ROOT}/.agents/skills/training-queue/scripts/training_queue.sh"
MANIFEST_PATH="${OUTPUT_ROOT%/}/court_detection/query_consistency_ablation/manifest.json"
CANONICAL_DATA_ROOT="/home/kamimura/projects/tennis-lab/data"
CANONICAL_EXTERNAL_ROOT="/home/kamimura/projects/tennis-lab/third_party"

source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
source "${REPO_ROOT}/scripts/colab/setup/prepare_archive_dataset.sh"
source "${REPO_ROOT}/scripts/colab/setup/path_contract.sh"

validate_colab_role_root DATA_ROOT "${DATA_ROOT}"
validate_colab_role_root OUTPUT_ROOT "${OUTPUT_ROOT}"
validate_colab_role_root CHECKPOINT_ROOT "${CHECKPOINT_ROOT}"
validate_colab_role_root TRAINING_QUEUE_DIR "${TRAINING_QUEUE_DIR}"

if [[ ! -d /content/drive/MyDrive ]]; then
    echo "[issue790] Google Drive is not mounted at /content/drive/MyDrive." >&2
    exit 1
fi

install_colab_dependencies "${REPO_ROOT}"
DATA_DIR="${DATA_ROOT}" prepare_archive_dataset court_query_issue790_v3 "${REPO_ROOT}"

DATASET_ROOT="${DATA_ROOT%/}/issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes/B00/datasets/court"
DERIVED_ROOT="${DATA_ROOT%/}/court_detection/derived_targets_issue790_v3"
BACKBONE_CKPT="${REPO_ROOT}/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
if [[ ! -f "${DATASET_ROOT}/dataset.json" || ! -d "${DERIVED_ROOT}" ]]; then
    echo "[issue790] V3 dataset/derived targets are missing after archive staging." >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" ]]; then
    echo "[issue790] DINOv3 source/checkpoint is missing after archive staging." >&2
    exit 1
fi

cd "${REPO_ROOT}"
RUNTIME_PYTHON="$(command -v python)"
mkdir -p "$(dirname "${MANIFEST_PATH}")"
echo "[issue790] generating canonical v3 manifest: ${MANIFEST_PATH}"
"${RUNTIME_PYTHON}" -m src.tasks.court_detection.scripts.run_query_consistency_ablation \
    "paths.data_root=${CANONICAL_DATA_ROOT}" \
    "paths.external_asset_root=${CANONICAL_EXTERNAL_ROOT}" \
    "paths.output_root=${OUTPUT_ROOT}"

if [[ "${SHARD_NAME}" == "colab-1" ]]; then
    START_INDEX=0
    END_INDEX=12
else
    START_INDEX=12
    END_INDEX=24
fi

"${RUNTIME_PYTHON}" -m scripts.colab.setup.enqueue_query_consistency_shard \
    --manifest "${MANIFEST_PATH}" \
    --seed 42 \
    --job-kind train \
    --start-index "${START_INDEX}" \
    --end-index "${END_INDEX}" \
    --python-executable "${RUNTIME_PYTHON}" \
    --data-root "${DATA_ROOT}" \
    --external-asset-root "${REPO_ROOT}/third_party" \
    --output-root "${OUTPUT_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --queue-script "${QUEUE_SCRIPT}" \
    --queue-dir "${TRAINING_QUEUE_DIR}" \
    --repository-root "${REPO_ROOT}" \
    --provider colab \
    --session "${SESSION_ID}" \
    --issue 790

echo "[issue790] starting ${SHARD_NAME} grid jobs (${START_INDEX}-${END_INDEX})."
TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" \
TRAINING_QUEUE_PYTHON="${RUNTIME_PYTHON}" \
bash "${QUEUE_SCRIPT}" serve --idle-timeout 30

FINAL_QUEUE_STATUS="$(TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" bash "${QUEUE_SCRIPT}" status)"
echo "${FINAL_QUEUE_STATUS}"
if [[ "${FINAL_QUEUE_STATUS}" != *"worker: stopped"* \
      || "${FINAL_QUEUE_STATUS}" != *"queued=0 running=0 done=12 failed=0"* ]]; then
    echo "[issue790] grid shard failed; inspect ${TRAINING_QUEUE_DIR}/logs." >&2
    exit 1
fi
echo "[issue790] ${SHARD_NAME} grid finished."
