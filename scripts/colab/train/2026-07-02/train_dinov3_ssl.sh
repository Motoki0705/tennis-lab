#!/usr/bin/env bash
set -euo pipefail

# Train DINOv3 ViT-B/16 LoRA SSL on the tennis-court YouTube image corpus (Colab).
#
# Mirrors the local run validated in issue #524 / #579
# (data/tennis/dino_ssl/images, 298,300 imgs; see
# knowledge/nodes/run-i524-ssl-lora-vitb16.md and
# outputs/dino_ssl/vitb16_lora_r64_e8_bs16/run_watchdog.sh): the SSL meta-arch
# injects LoRA adapters into the student/EMA-teacher backbones while freezing
# the pretrained ViT-B/16 weights, keeping the existing DINO/iBOT training
# pipeline unchanged.
#
# Responsibilities:
#   - install dependencies and stage the DINOv3 submodule, base checkpoint,
#     and SSL image corpus
#   - run LoRA SSL training with checkpoints/logs written to Google Drive
#
# dinov3/train/train.py resumes from the latest checkpoint in --output-dir by
# default (pass --no-resume via extra args to disable), so re-running this
# script after a Colab disconnect just continues training.
#
# Usage from Colab (after mounting Google Drive):
#   !bash scripts/colab/train/2026-07-02/train_dinov3_ssl.sh
#
# Config/architecture experiments: append DINOv3 config overrides, e.g.
#   !bash scripts/colab/train/2026-07-02/train_dinov3_ssl.sh lora.rank=32 lora.alpha=64
#   !bash scripts/colab/train/2026-07-02/train_dinov3_ssl.sh optim.epochs=4
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path
#   DINOV3_ROOT   default: ${REPO_ROOT}/third_party/dinov3
#   CONFIG_FILE   default: dinov3/configs/train/dinov3_vitb16_lora.yaml (relative to DINOV3_ROOT)
#   DATASET_DIR   default: ${REPO_ROOT}/data/tennis/dino_ssl/images
#   OUTPUT_DIR    default: /content/drive/MyDrive/tennis_lab/outputs/dino_ssl/vitb16_lora
#   NUM_WORKERS   default: 2 (a prior local WSL2 run leaked host RAM per
#                 DataLoader worker at higher values; raise only once the
#                 Colab runtime is confirmed stable at that setting)

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DINOV3_ROOT="${DINOV3_ROOT:-${REPO_ROOT}/third_party/dinov3}"
CONFIG_FILE="${CONFIG_FILE:-dinov3/configs/train/dinov3_vitb16_lora.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/tennis/dino_ssl/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/dino_ssl/vitb16_lora}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CKPT_NAME="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
CKPT_PATH="${DINOV3_ROOT}/checkpoints/${CKPT_NAME}"

source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
source "${REPO_ROOT}/scripts/colab/setup/prepare_archive_dataset.sh"
install_colab_dependencies "${REPO_ROOT}"
prepare_archive_dataset dinov3_ssl "${REPO_ROOT}"

echo "[train_dinov3_ssl] repo root: ${REPO_ROOT}"
echo "[train_dinov3_ssl] dinov3 root: ${DINOV3_ROOT}"
echo "[train_dinov3_ssl] dataset dir: ${DATASET_DIR}"
echo "[train_dinov3_ssl] output dir: ${OUTPUT_DIR}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_dinov3_ssl] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_dinov3_ssl] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

if [[ ! -d "${DINOV3_ROOT}/dinov3" || ! -f "${CKPT_PATH}" || ! -d "${DATASET_DIR}" || -z "$(ls -A "${DATASET_DIR}" 2>/dev/null)" ]]; then
    echo "[train_dinov3_ssl] missing DINOv3 submodule, checkpoint, or SSL image corpus." >&2
    echo "[train_dinov3_ssl] setup completed without producing the required assets." >&2
    exit 1
fi

cd "${DINOV3_ROOT}"

echo "[train_dinov3_ssl] starting training (extra overrides: ${*:-none})"
PYTHONPATH="${DINOV3_ROOT}" python dinov3/train/train.py \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    "train.dataset_path=ImageDirectory:root=${DATASET_DIR}" \
    "train.num_workers=${NUM_WORKERS}" \
    "$@"

echo "[train_dinov3_ssl] done."
