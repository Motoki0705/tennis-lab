#!/usr/bin/env bash
set -euo pipefail

# Train monocular-broadcast PLCS on Colab (issue #593 recipe).
#
# Defaults reproduce run-mono3d-plcs-bcast (knowledge/nodes/run-mono3d-plcs-bcast.md):
#   multiview_axial_split (H=0 / S=6) + canonical_rot (aux weights 0,
#   position_weight=8.0) + chunked_singleview_sequence (C=1) + camera=broadcast
#   + num_court_kp=14, 200 epochs.
#
# Responsibilities:
#   - verify SMPL-H / ACCAD assets exist (needed by background chunk generation)
#   - generate data/plcs_broadcast (1000 scenes) if it does not exist yet
#   - run training with checkpoints/logs written to Google Drive
#
# Usage from Colab (after scripts/colab/setup/install_deps.sh and
# scripts/colab/setup/prepare_archive_dataset.sh plcs):
#   !bash scripts/colab/train/train_plcs_broadcast.sh
#
# Architecture experiments: append Hydra overrides, e.g.
#   !bash scripts/colab/train/train_plcs_broadcast.sh model.num_task_layers=8
#   !bash scripts/colab/train/train_plcs_broadcast.sh data.seq_len_range=[128,512]
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path
#   DATASET_DIR   default: ${REPO_ROOT}/data/plcs_broadcast
#   OUTPUT_DIR    default: /content/drive/MyDrive/tennis_lab/outputs/plcs_broadcast
#   NUM_SCENES    default: 1000 (only used when generating the dataset)
#   NUM_WORKERS   default: 2  (dataloader workers; raise on A100/L4 instances)
#   GEN_WORKERS   default: 4  (scene/chunk generation workers)
#   MAX_EPOCHS    default: 200

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/plcs_broadcast}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/plcs_broadcast}"
NUM_SCENES="${NUM_SCENES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GEN_WORKERS="${GEN_WORKERS:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

echo "[train_plcs_broadcast] repo root: ${REPO_ROOT}"
echo "[train_plcs_broadcast] dataset dir: ${DATASET_DIR}"
echo "[train_plcs_broadcast] output dir: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_plcs_broadcast] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_plcs_broadcast] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

# Chunked training keeps generating fresh scenes in the background, so the
# SMPL-H model and ACCAD motions are required at train time, not only for the
# initial dataset generation.
for asset in data/smplx/smplh data/ACCAD; do
    if [[ ! -d "${REPO_ROOT}/${asset}" ]]; then
        echo "[train_plcs_broadcast] missing asset: ${REPO_ROOT}/${asset}" >&2
        echo "[train_plcs_broadcast] Run this first:" >&2
        echo "  bash scripts/colab/setup/prepare_archive_dataset.sh plcs" >&2
        exit 1
    fi
done

if [[ ! -f "${DATASET_DIR}/meta.json" ]]; then
    echo "[train_plcs_broadcast] dataset not found. Generating ${NUM_SCENES} broadcast scenes..."
    python -m src.tasks.plcs.scripts.generate_dataset \
        camera=broadcast \
        "run.output_dir=${DATASET_DIR}" \
        "simulation.num_scenes=${NUM_SCENES}" \
        "run.num_workers=${GEN_WORKERS}"
else
    echo "[train_plcs_broadcast] dataset already exists: ${DATASET_DIR}"
fi

echo "[train_plcs_broadcast] starting training (extra overrides: ${*:-none})"
python -m src.tasks.plcs.scripts.train \
    model=multiview_axial_split \
    model.num_layers=0 \
    model.num_task_layers=6 \
    data=chunked_singleview_sequence \
    "data.scene_dir=${DATASET_DIR}" \
    "data.chunk.chunks_dir=${DATASET_DIR}/chunks" \
    data.batch_size=8 \
    "data.seq_len_range=[64,256]" \
    data.num_court_kp=14 \
    "data.num_workers=${NUM_WORKERS}" \
    "data.chunk.generation_workers=${GEN_WORKERS}" \
    loss=canonical_rot \
    loss.position_weight=8.0 \
    loss.canonical_pose_weight=0.0 \
    loss.joint_angle_weight=0.0 \
    loss.torsion_angle_weight=0.0 \
    loss.torso_twist_weight=0.0 \
    loss.bone_length_weight=0.0 \
    training.trainer.accumulate_grad_batches=1 \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.early_stopping.enabled=false \
    camera=broadcast \
    "run.output_dir=${OUTPUT_DIR}" \
    run.gpus=1 \
    "$@"

echo "[train_plcs_broadcast] done."
