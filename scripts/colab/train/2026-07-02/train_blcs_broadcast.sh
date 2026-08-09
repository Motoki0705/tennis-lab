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
#   REPO_ROOT     default: repository root inferred from this script path
#   DATASET_DIR   default: ${REPO_ROOT}/data/blcs_broadcast
#   OUTPUT_DIR    default: /content/drive/MyDrive/tennis_lab/outputs/blcs_broadcast
#   NUM_SCENES    default: 1000 (only used when generating the dataset)
#   NUM_WORKERS   default: 2  (dataloader workers; raise on A100/L4 instances)
#   GEN_WORKERS   default: 4  (scene/chunk generation workers)
#   MAX_EPOCHS    default: 200
#   EPOCHS_PER_CHUNK  default: 20 (long chunk reuse minimizes generation stalls)

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/blcs_broadcast}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/blcs_broadcast}"
NUM_SCENES="${NUM_SCENES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GEN_WORKERS="${GEN_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-20}"

source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
source "${REPO_ROOT}/scripts/colab/setup/prepare_generated_dataset.sh"
install_colab_dependencies "${REPO_ROOT}"
prepare_generated_dataset blcs "${REPO_ROOT}" "${DATASET_DIR}" \
    camera=broadcast \
    "generator.num_scenes=${NUM_SCENES}" \
    "run.num_workers=${GEN_WORKERS}"

echo "[train_blcs_broadcast] repo root: ${REPO_ROOT}"
echo "[train_blcs_broadcast] dataset dir: ${DATASET_DIR}"
echo "[train_blcs_broadcast] output dir: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_blcs_broadcast] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_blcs_broadcast] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

echo "[train_blcs_broadcast] starting training (extra overrides: ${*:-none})"
python -m src.tasks.blcs.scripts.train \
    --config-name train_chunked \
    camera=broadcast \
    "data.scene_dir=${DATASET_DIR}" \
    "data.chunk.chunks_dir=${DATASET_DIR}/chunks" \
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
