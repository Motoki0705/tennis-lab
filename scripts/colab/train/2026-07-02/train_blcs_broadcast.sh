#!/usr/bin/env bash
set -euo pipefail

# Train monocular-broadcast BLCS on Colab (issue #593 recipe).
#
# Defaults reproduce run-mono3d-blcs-bcast-v2-yw4
# (knowledge/nodes/run-mono3d-blcs-bcast-v2-yw4.md) with max_epochs extended
# 100 -> 200, since v2 had not converged at 100 epochs:
#   train_chunked (multiview_axial_base) + num_views_range=[1,1] (C=1)
#   + camera=broadcast + num_court_kp=14
#   + training.position_axis_weights=[1,4,1] (monocular depth-axis fix).
#
# Responsibilities:
#   - install the common Colab dependencies
#   - generate data/blcs_broadcast (1000 scenes) if it does not exist yet
#     (BLCS scenes are pure physics simulation; no external assets needed)
#   - run training with checkpoints/logs written to Google Drive
#
# Usage from Colab (after mounting Google Drive):
#   !bash scripts/colab/train/2026-07-02/train_blcs_broadcast.sh
#
# Architecture experiments: append Hydra overrides, e.g.
#   !bash scripts/colab/train/2026-07-02/train_blcs_broadcast.sh data.seq_len_range=[128,384]
#   !bash scripts/colab/train/2026-07-02/train_blcs_broadcast.sh training.reprojection_loss_weight=0.3
#
# Environment overrides:
#   REPO_ROOT       default: repository root inferred from this script path
#   DATA_ROOT       default: ${REPO_ROOT}/data (absolute Hydra data root)
#   DATASET_DIR     default: blcs_broadcast (DATA_ROOT-relative)
#   ARTIFACT_ROOT   default: ${DATA_ROOT} (absolute Hydra artifact root)
#   CHUNKS_DIR      default: ${DATASET_DIR}/chunks (ARTIFACT_ROOT-relative)
#   OUTPUT_ROOT     default: /content/drive/MyDrive/tennis_lab/outputs
#   OUTPUT_DIR      default: blcs_broadcast (OUTPUT_ROOT-relative)
#   CHECKPOINT_ROOT default: ${OUTPUT_ROOT} (absolute Hydra checkpoint root)
#   NUM_SCENES      default: 1000 (only used when generating the dataset)
#   NUM_WORKERS     default: 2  (dataloader workers; raise on A100/L4 instances)
#   GEN_WORKERS     default: 4  (scene/chunk generation workers)
#   MAX_EPOCHS      default: 200
#   EPOCHS_PER_CHUNK  default: 20 (long chunk reuse minimizes generation stalls)

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_DIR="${DATASET_DIR:-blcs_broadcast}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DATA_ROOT}}"
CHUNKS_DIR="${CHUNKS_DIR:-${DATASET_DIR}/chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-blcs_broadcast}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}}"
NUM_SCENES="${NUM_SCENES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GEN_WORKERS="${GEN_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-20}"

source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
source "${REPO_ROOT}/scripts/colab/setup/prepare_generated_dataset.sh"
validate_colab_role_root DATA_ROOT "${DATA_ROOT}"
validate_colab_role_child DATASET_DIR "${DATASET_DIR}"
validate_colab_role_root ARTIFACT_ROOT "${ARTIFACT_ROOT}"
validate_colab_role_child CHUNKS_DIR "${CHUNKS_DIR}"
validate_colab_role_root OUTPUT_ROOT "${OUTPUT_ROOT}"
validate_colab_role_child OUTPUT_DIR "${OUTPUT_DIR}"
validate_colab_role_root CHECKPOINT_ROOT "${CHECKPOINT_ROOT}"

DATASET_PATH="${DATA_ROOT%/}/${DATASET_DIR}"
OUTPUT_PATH="${OUTPUT_ROOT%/}/${OUTPUT_DIR}"

echo "[train_blcs_broadcast] repo root: ${REPO_ROOT}"
echo "[train_blcs_broadcast] dataset path: ${DATASET_PATH}"
echo "[train_blcs_broadcast] output path: ${OUTPUT_PATH}"

if [[ ( "${OUTPUT_ROOT}" == /content/drive/* || "${CHECKPOINT_ROOT}" == /content/drive/* ) \
      && ! -d /content/drive/MyDrive ]]; then
    echo "[train_blcs_broadcast] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_blcs_broadcast] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

install_colab_dependencies "${REPO_ROOT}"
prepare_generated_dataset blcs "${REPO_ROOT}" "${DATA_ROOT}" "${DATASET_DIR}" \
    camera=broadcast \
    "generator.num_scenes=${NUM_SCENES}" \
    "run.num_workers=${GEN_WORKERS}"

cd "${REPO_ROOT}"
echo "[train_blcs_broadcast] starting training (extra overrides: ${*:-none})"
python -m src.tasks.blcs.scripts.train \
    --config-name train_chunked \
    camera=broadcast \
    model.num_court_tokens=14 \
    "paths.data_root=${DATA_ROOT}" \
    "paths.artifact_root=${ARTIFACT_ROOT}" \
    "paths.output_root=${OUTPUT_ROOT}" \
    "paths.checkpoint_root=${CHECKPOINT_ROOT}" \
    "data.scene_dir=${DATASET_DIR}" \
    "data.chunk.chunks_dir=${CHUNKS_DIR}" \
    "data.num_views_range=[1,1]" \
    data.camera_mode=random \
    data.num_court_kp=14 \
    "data.num_workers=${NUM_WORKERS}" \
    "data.chunk.generation_workers=${GEN_WORKERS}" \
    "data.chunk.epochs_per_chunk=${EPOCHS_PER_CHUNK}" \
    "training.position_axis_weights=[1.0,4.0,1.0]" \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.trainer.check_val_every_n_epoch=5 \
    training.qualitative_logging.enabled=false \
    "run.output_dir=${OUTPUT_DIR}" \
    run.gpus=1 \
    "$@"

echo "[train_blcs_broadcast] done."
