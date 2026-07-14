#!/usr/bin/env bash
set -euo pipefail

# Train the DINOv3+DPT court white-line segmentation model on Colab with a LoRA
# backbone, switchable between the original and the tennis-SSL DINOv3 ViT-B/16
# weights to measure how much the extra SSL changes downstream accuracy.
#
# Uses the unified court-detection pipeline with data=court_line, one output
# channel, and the line BCE+Dice loss. Checkpoints and early stopping are
# selected by val/dice (max), the task metric for binary line segmentation.
#
# Responsibilities:
#   - verify DINOv3 assets, the court dataset, and line masks are staged
#     (run scripts/colab/setup/prepare_archive_dataset.sh court first)
#   - run training with checkpoints/logs written to Google Drive
#   - auto-resume: if a last.ckpt exists on Drive, training resumes
#     full-state after a Colab disconnect
#
# Usage from Colab (after scripts/colab/setup/install_deps.sh and
# scripts/colab/setup/prepare_archive_dataset.sh court):
#   !bash scripts/colab/train/train_court_line_dinov3_dpt.sh orig
#   !bash scripts/colab/train/train_court_line_dinov3_dpt.sh ssl
#
# Extra Hydra overrides are appended, e.g.:
#   !bash scripts/colab/train/train_court_line_dinov3_dpt.sh ssl data.num_workers=2
#
# Environment overrides:
#   REPO_ROOT    default: repository root inferred from this script path
#   OUTPUT_DIR   default: /content/drive/MyDrive/tennis_lab/outputs/court_detection/dinov3_dpt_line_${VARIANT}
#   MAX_EPOCHS   default: 20
#   BATCH_SIZE   default: 8

usage() {
    echo "Usage: bash scripts/colab/train/train_court_line_dinov3_dpt.sh {orig|ssl} [hydra overrides...]" >&2
}

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
    usage
    exit 2
fi
shift

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/court_detection/dinov3_dpt_line_${VARIANT}}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"

CKPT_DIR="${REPO_ROOT}/third_party/dinov3/checkpoints"
COURT_DATA_DIR="${REPO_ROOT}/data/court"
case "${VARIANT}" in
    orig)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ;;
    ssl)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_tennis_ssl_merged.pth"
        ;;
    *)
        echo "[train_court_line_dinov3_dpt] unknown variant: ${VARIANT}" >&2
        usage
        exit 2
        ;;
esac

echo "[train_court_line_dinov3_dpt] repo root: ${REPO_ROOT}"
echo "[train_court_line_dinov3_dpt] variant: ${VARIANT}"
echo "[train_court_line_dinov3_dpt] backbone: ${BACKBONE_CKPT}"
echo "[train_court_line_dinov3_dpt] output dir: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_court_line_dinov3_dpt] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_court_line_dinov3_dpt] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" \
      || ! -d "${COURT_DATA_DIR}" || ! -d "${COURT_DATA_DIR}/line_masks" ]]; then
    echo "[train_court_line_dinov3_dpt] missing DINOv3 assets, court dataset, or line masks" >&2
    echo "[train_court_line_dinov3_dpt] (need third_party/dinov3, ${BACKBONE_CKPT}, data/court, and data/court/line_masks)." >&2
    echo "[train_court_line_dinov3_dpt] Run this first:" >&2
    echo "  bash scripts/colab/setup/prepare_archive_dataset.sh court" >&2
    exit 1
fi

# Colab disconnect recovery: resume from the newest version's last.ckpt.
# (The base runner's TensorBoardLogger auto-increments logs/version_N.)
RESUME_OVERRIDES=()
LAST_CKPTS=("${OUTPUT_DIR}"/logs/version_*/checkpoints/last.ckpt)
if [[ -f "${LAST_CKPTS[-1]}" ]]; then
    RESUME_CKPT="$(printf '%s\n' "${LAST_CKPTS[@]}" | sort -V | tail -1)"
    echo "[train_court_line_dinov3_dpt] resuming from ${RESUME_CKPT}"
    RESUME_OVERRIDES=("run.resume=${RESUME_CKPT}")
fi

echo "[train_court_line_dinov3_dpt] starting training (extra overrides: ${*:-none})"
python -m src.tasks.court_detection.scripts.train \
    data=court_line \
    "data.augmentation.train_scales=[256]" \
    data.augmentation.val_short_side=256 \
    model/encoder=dinov3 \
    model/decoder=dpt \
    model.name=dinov3_dpt \
    model.num_classes=1 \
    training=lora \
    loss=line \
    "model.encoder.checkpoint_path=${BACKBONE_CKPT}" \
    "data.batch_size=${BATCH_SIZE}" \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.checkpoint.monitor=val/dice \
    training.checkpoint.mode=max \
    training.early_stopping.monitor=val/dice \
    training.early_stopping.mode=max \
    "run.output_dir=${OUTPUT_DIR}" \
    "${RESUME_OVERRIDES[@]}" \
    "$@"

echo "[train_court_line_dinov3_dpt] done."
