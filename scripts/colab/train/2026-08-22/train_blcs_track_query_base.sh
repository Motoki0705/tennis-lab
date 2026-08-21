#!/usr/bin/env bash
set -euo pipefail

# Train the BLCS base-size multi-ball track-query model on Colab.
#
# PR #769 classifies multi-ball tracking as a separate contract from the
# deployed single-ball BLCS model. This script uses that contract's current
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
#   !bash scripts/colab/train/2026-08-22/train_blcs_track_query_base.sh
#
# Resume explicitly from a Drive checkpoint when needed:
#   !bash scripts/colab/train/2026-08-22/train_blcs_track_query_base.sh \
#       run.resume=/content/drive/MyDrive/tennis_lab/outputs/blcs_track_query_base/logs/version_0/checkpoints/last.ckpt
#
# Environment overrides:
#   REPO_ROOT         default: repository root inferred from this script path
#   DATASET_DIR       default: ${REPO_ROOT}/data/blcs/multi_object_lifecycle_colab
#   OUTPUT_DIR        default: /content/drive/MyDrive/tennis_lab/outputs/blcs_track_query_base
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
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/blcs/multi_object_lifecycle_colab}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/blcs_track_query_base}"
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
source "${REPO_ROOT}/scripts/colab/setup/prepare_generated_dataset.sh"
install_colab_dependencies "${REPO_ROOT}"
prepare_generated_dataset blcs "${REPO_ROOT}" "${DATASET_DIR}" \
    generation=multi_object \
    "generator.num_scenes=${NUM_SCENES}" \
    "run.num_workers=${GEN_WORKERS}"

echo "[train_blcs_track_query_base] repo root: ${REPO_ROOT}"
echo "[train_blcs_track_query_base] dataset dir: ${DATASET_DIR}"
echo "[train_blcs_track_query_base] output dir: ${OUTPUT_DIR}"
echo "[train_blcs_track_query_base] batch size: ${BATCH_SIZE}"
echo "[train_blcs_track_query_base] CSWA backend: ${CSWA_BACKEND}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_blcs_track_query_base] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_blcs_track_query_base] Mount it before training so checkpoints survive VM resets." >&2
    exit 1
fi

echo "[train_blcs_track_query_base] starting training (extra overrides: ${*:-none})"
MPLBACKEND="${MPLBACKEND:-Agg}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python -m src.tasks.blcs.scripts.train \
    --config-name train_tracking_chunked \
    model=track_query_base \
    "model.cswa.backend=${CSWA_BACKEND}" \
    "data.scene_dir=${DATASET_DIR}" \
    "data.chunk.chunks_dir=${DATASET_DIR}/chunks" \
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

echo "[train_blcs_track_query_base] done."
