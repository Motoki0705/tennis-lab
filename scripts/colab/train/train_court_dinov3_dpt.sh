#!/usr/bin/env bash
set -euo pipefail

# Train the DINOv3+DPT court keypoint detector (14 KPs) on Colab with a LoRA
# backbone, switchable between the original and the tennis-SSL DINOv3 ViT-B/16
# weights to measure how much the extra SSL changes downstream accuracy.
#
# Reproduces the established dinov3_dpt court-KP baseline command
# (data=court_kp, train_scales=[256], training=lora, loss=kp, batch 8) with
# max_epochs 20 and the issue #618 recipe on top: checkpoints and early
# stopping are selected by the task metric val/mean_dist (min) instead of
# val/loss, which was shown to decouple from the task metric
# (knowledge/nodes/run-i618-convnext-v2-ft.md).
#
# Responsibilities:
#   - verify DINOv3 assets and the court dataset are staged
#     (run scripts/colab/setup/prepare_archive_dataset.sh court first)
#   - run training with checkpoints/logs written to Google Drive
#   - auto-resume: if a last.ckpt exists on Drive, training resumes
#     full-state after a Colab disconnect
#
# Usage from Colab (after scripts/colab/setup/install_deps.sh and
# scripts/colab/setup/prepare_archive_dataset.sh court):
#   !bash scripts/colab/train/train_court_dinov3_dpt.sh orig
#   !bash scripts/colab/train/train_court_dinov3_dpt.sh ssl
#
# Extra Hydra overrides are appended, e.g.:
#   !bash scripts/colab/train/train_court_dinov3_dpt.sh ssl data.num_workers=2
#
# Environment overrides:
#   REPO_ROOT    default: repository root inferred from this script path
#   OUTPUT_DIR   default: /content/drive/MyDrive/tennis_lab/outputs/court_detection/dinov3_dpt_${VARIANT}
#   MAX_EPOCHS   default: 20
#   BATCH_SIZE   default: 8

usage() {
    echo "Usage: bash scripts/colab/train/train_court_dinov3_dpt.sh {orig|ssl} [hydra overrides...]" >&2
}

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
    usage
    exit 2
fi
shift

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/content/drive/MyDrive/tennis_lab/outputs/court_detection/dinov3_dpt_${VARIANT}}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"

CKPT_DIR="${REPO_ROOT}/third_party/dinov3/checkpoints"
case "${VARIANT}" in
    orig)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ;;
    ssl)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_tennis_ssl_merged.pth"
        ;;
    *)
        echo "[train_court_dinov3_dpt] unknown variant: ${VARIANT}" >&2
        usage
        exit 2
        ;;
esac

echo "[train_court_dinov3_dpt] repo root: ${REPO_ROOT}"
echo "[train_court_dinov3_dpt] variant: ${VARIANT}"
echo "[train_court_dinov3_dpt] backbone: ${BACKBONE_CKPT}"
echo "[train_court_dinov3_dpt] output dir: ${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_DIR}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_court_dinov3_dpt] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_court_dinov3_dpt] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" \
      || ! -d "${REPO_ROOT}/data/court" ]]; then
    echo "[train_court_dinov3_dpt] missing DINOv3 assets or court dataset" >&2
    echo "[train_court_dinov3_dpt] (need third_party/dinov3, ${BACKBONE_CKPT} and data/court)." >&2
    echo "[train_court_dinov3_dpt] Run this first:" >&2
    echo "  bash scripts/colab/setup/prepare_archive_dataset.sh court" >&2
    exit 1
fi

# Colab disconnect recovery: resume from the newest version's last.ckpt.
# (The base runner's TensorBoardLogger auto-increments logs/version_N.)
RESUME_OVERRIDES=()
LAST_CKPTS=("${OUTPUT_DIR}"/logs/version_*/checkpoints/last.ckpt)
if [[ -f "${LAST_CKPTS[-1]}" ]]; then
    RESUME_CKPT="$(printf '%s\n' "${LAST_CKPTS[@]}" | sort -V | tail -1)"
    echo "[train_court_dinov3_dpt] resuming from ${RESUME_CKPT}"
    RESUME_OVERRIDES=("run.resume=${RESUME_CKPT}")
fi

echo "[train_court_dinov3_dpt] starting training (extra overrides: ${*:-none})"
python -m src.tasks.court_detection.scripts.train \
    data=court_kp \
    "data.augmentation.train_scales=[256]" \
    data.augmentation.val_short_side=256 \
    model/encoder=dinov3 \
    model/decoder=dpt \
    model.name=dinov3_dpt \
    model.num_classes=14 \
    training=lora \
    loss=kp \
    "model.encoder.checkpoint_path=${BACKBONE_CKPT}" \
    "data.batch_size=${BATCH_SIZE}" \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.checkpoint.monitor=val/mean_dist \
    training.checkpoint.mode=min \
    training.early_stopping.monitor=val/mean_dist \
    training.early_stopping.mode=min \
    "run.output_dir=${OUTPUT_DIR}" \
    "${RESUME_OVERRIDES[@]}" \
    "$@"

echo "[train_court_dinov3_dpt] done."
