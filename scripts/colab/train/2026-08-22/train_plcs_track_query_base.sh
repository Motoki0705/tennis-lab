#!/usr/bin/env bash
set -euo pipefail

# Train the PLCS base-size multi-person track-query model on Colab.
#
# PR #769 classifies multi-person tracking as a separate contract from the
# deployed single-person PLCS model. This script uses that contract's current
# lifecycle path: fixed-Q track queries, multi-object scenes, 3-5 views, and
# 512-1024-frame clips. Train scenes are generated in background chunks while
# validation and test use the fixed dataset generated before training.
#
# Fixed recipe defaults:
#   - model=track_query_base (D=512, H=8, 8 stages)
#   - train_tracking_chunked
#   - max_epochs=200
#   - physical/effective batch size=4 (no gradient accumulation)
#   - 1000 scenes per chunk, 20 epochs per chunk
#
# Usage from Colab after mounting Google Drive:
#   !bash scripts/colab/train/2026-08-22/train_plcs_track_query_base.sh
#
# Resume explicitly from a Drive checkpoint when needed:
#   !bash scripts/colab/train/2026-08-22/train_plcs_track_query_base.sh \
#       run.resume=plcs_track_query_base/logs/version_0/checkpoints/last.ckpt
# `run.resume` is relative to CHECKPOINT_ROOT (OUTPUT_ROOT by default).
#
# Environment overrides:
#   REPO_ROOT         default: repository root inferred from this script path
#   DATA_ROOT         default: ${REPO_ROOT}/data (absolute Hydra data root)
#   DATASET_DIR       default: plcs/multi_object_lifecycle_colab (DATA_ROOT-relative)
#   ARTIFACT_ROOT     default: ${DATA_ROOT} (absolute Hydra artifact root)
#   CHUNKS_DIR        default: ${DATASET_DIR}/chunks (ARTIFACT_ROOT-relative)
#   OUTPUT_ROOT       default: /content/drive/MyDrive/tennis_lab/outputs
#   OUTPUT_DIR        default: plcs_track_query_base (OUTPUT_ROOT-relative)
#   CHECKPOINT_ROOT   default: ${OUTPUT_ROOT} (absolute Hydra checkpoint root)
#   NUM_SCENES        default: 1000 fixed train/val/test scenes
#   NUM_WORKERS       default: 2 DataLoader workers
#   GEN_WORKERS       default: 4 dataset/chunk generation workers
#   SCENES_PER_CHUNK  default: 1000
#   EPOCHS_PER_CHUNK  default: 20
#   PREFETCH_CHUNKS   default: 5
#   MAX_EPOCHS        default: 200
#   BATCH_SIZE        default: 4
#   CSWA_BACKEND      default: cuda; use reference only when chosen explicitly

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_DIR="${DATASET_DIR:-plcs/multi_object_lifecycle_colab}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${DATA_ROOT}}"
CHUNKS_DIR="${CHUNKS_DIR:-${DATASET_DIR}/chunks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-plcs_track_query_base}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}}"
NUM_SCENES="${NUM_SCENES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GEN_WORKERS="${GEN_WORKERS:-4}"
SCENES_PER_CHUNK="${SCENES_PER_CHUNK:-1000}"
EPOCHS_PER_CHUNK="${EPOCHS_PER_CHUNK:-20}"
PREFETCH_CHUNKS="${PREFETCH_CHUNKS:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CSWA_BACKEND="${CSWA_BACKEND:-cuda}"

source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
source "${REPO_ROOT}/scripts/colab/setup/prepare_archive_dataset.sh"
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

echo "[train_plcs_track_query_base] repo root: ${REPO_ROOT}"
echo "[train_plcs_track_query_base] dataset path: ${DATASET_PATH}"
echo "[train_plcs_track_query_base] output path: ${OUTPUT_PATH}"
echo "[train_plcs_track_query_base] batch size: ${BATCH_SIZE}"
echo "[train_plcs_track_query_base] CSWA backend: ${CSWA_BACKEND}"

if [[ ( "${OUTPUT_ROOT}" == /content/drive/* || "${CHECKPOINT_ROOT}" == /content/drive/* ) \
      && ! -d /content/drive/MyDrive ]]; then
    echo "[train_plcs_track_query_base] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_plcs_track_query_base] Mount it before training so checkpoints survive VM resets." >&2
    exit 1
fi

install_colab_dependencies "${REPO_ROOT}"
prepare_archive_dataset plcs "${REPO_ROOT}"
prepare_generated_dataset plcs "${REPO_ROOT}" "${DATA_ROOT}" "${DATASET_DIR}" \
    generation=multi_object \
    "simulation.num_scenes=${NUM_SCENES}" \
    "run.num_workers=${GEN_WORKERS}"

# Chunk generation needs these assets for the full duration of training.
for asset in data/smplx/smplh data/ACCAD; do
    if [[ ! -d "${REPO_ROOT}/${asset}" ]]; then
        echo "[train_plcs_track_query_base] missing asset: ${REPO_ROOT}/${asset}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
echo "[train_plcs_track_query_base] starting training (extra overrides: ${*:-none})"
MPLBACKEND="${MPLBACKEND:-Agg}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python -m src.tasks.plcs.scripts.train \
    --config-name train_tracking_chunked \
    model=track_query_base \
    "model.cswa.backend=${CSWA_BACKEND}" \
    "paths.data_root=${DATA_ROOT}" \
    "paths.artifact_root=${ARTIFACT_ROOT}" \
    "paths.output_root=${OUTPUT_ROOT}" \
    "paths.checkpoint_root=${CHECKPOINT_ROOT}" \
    "data.scene_dir=${DATASET_DIR}" \
    "data.chunk.chunks_dir=${CHUNKS_DIR}" \
    "data.batch_size=${BATCH_SIZE}" \
    "data.num_workers=${NUM_WORKERS}" \
    "data.chunk.scenes_per_chunk=${SCENES_PER_CHUNK}" \
    "data.chunk.generation_workers=${GEN_WORKERS}" \
    "data.chunk.epochs_per_chunk=${EPOCHS_PER_CHUNK}" \
    "data.chunk.prefetch_chunks=${PREFETCH_CHUNKS}" \
    training.trainer.accumulate_grad_batches=1 \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.trainer.check_val_every_n_epoch=5 \
    training.early_stopping.enabled=false \
    "run.output_dir=${OUTPUT_DIR}" \
    run.gpus=1 \
    "$@"

echo "[train_plcs_track_query_base] done."
