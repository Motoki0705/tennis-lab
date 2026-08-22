#!/usr/bin/env bash
set -euo pipefail

# Train the 14-channel Court KP target on the canonical synthetic schema-v2
# B00 dataset with a DINOv3 ViT-B/16 encoder, DPT decoder, and LoRA adapters.
#
# The script stages synthetic_court_v2.tar.zst and the original DINOv3
# checkpoint from Drive, writes checkpoints/logs back to Drive, and resumes
# full training state from the latest last.ckpt after a disconnect. The source
# schema, scene, target, encoder, and decoder are explicit Hydra overrides so
# this entry point cannot silently select v1 or a dense Court target.
#
# Usage from Colab (after mounting Google Drive):
#   !bash scripts/colab/train/2026-08-22/train_court_synthetic_v2_kp_dinov3_dpt.sh
#
# Extra Hydra overrides are appended, for example:
#   !bash scripts/colab/train/2026-08-22/train_court_synthetic_v2_kp_dinov3_dpt.sh data.num_workers=2
#
# Environment overrides:
#   REPO_ROOT   default: repository root inferred from this script path
#   DATA_ROOT   default: ${REPO_ROOT}/data (absolute Hydra data root)
#   OUTPUT_ROOT default: /content/drive/MyDrive/tennis_lab/outputs
#   OUTPUT_DIR  default: court_detection/synthetic_v2_kp_dinov3_dpt (OUTPUT_ROOT-relative)
#   CHECKPOINT_ROOT default: ${OUTPUT_ROOT} (absolute Hydra checkpoint root)
#   MAX_EPOCHS  default: 20
#   BATCH_SIZE  default: 8
#   INPUT_SIZE  default: 288

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-court_detection/synthetic_v2_kp_dinov3_dpt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INPUT_SIZE="${INPUT_SIZE:-288}"

BACKBONE_CKPT="${REPO_ROOT}/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DATASET_ROOT="${DATA_ROOT%/}/synthetic_data_generation/scenes/B00/datasets/court"
OUTPUT_PATH="${OUTPUT_ROOT%/}/${OUTPUT_DIR}"

# shellcheck source=scripts/colab/setup/install_deps.sh
source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
# shellcheck source=scripts/colab/setup/prepare_archive_dataset.sh
source "${REPO_ROOT}/scripts/colab/setup/prepare_archive_dataset.sh"
# shellcheck source=scripts/colab/setup/path_contract.sh
source "${REPO_ROOT}/scripts/colab/setup/path_contract.sh"
validate_colab_role_root DATA_ROOT "${DATA_ROOT}"
validate_colab_role_root OUTPUT_ROOT "${OUTPUT_ROOT}"
validate_colab_role_child OUTPUT_DIR "${OUTPUT_DIR}"
validate_colab_role_root CHECKPOINT_ROOT "${CHECKPOINT_ROOT}"

echo "[train_court_synthetic_v2] repo root: ${REPO_ROOT}"
echo "[train_court_synthetic_v2] dataset: ${DATASET_ROOT}"
echo "[train_court_synthetic_v2] backbone: ${BACKBONE_CKPT}"
echo "[train_court_synthetic_v2] output path: ${OUTPUT_PATH}"

if [[ ( "${OUTPUT_ROOT}" == /content/drive/* || "${CHECKPOINT_ROOT}" == /content/drive/* ) \
      && ! -d /content/drive/MyDrive ]]; then
    echo "[train_court_synthetic_v2] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_court_synthetic_v2] Mount it before training:" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

install_colab_dependencies "${REPO_ROOT}"
DATA_DIR="${DATA_ROOT}" prepare_archive_dataset synthetic_court_v2 "${REPO_ROOT}"

if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" ]]; then
    echo "[train_court_synthetic_v2] DINOv3 source or checkpoint is missing after setup." >&2
    exit 1
fi
if [[ ! -f "${DATASET_ROOT}/dataset.json" ]]; then
    echo "[train_court_synthetic_v2] synthetic Court dataset is missing after setup: ${DATASET_ROOT}" >&2
    exit 1
fi

DATASET_SCHEMA="$(python -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["schema"])' "${DATASET_ROOT}/dataset.json")"
if [[ "${DATASET_SCHEMA}" != "canonical_court_dataset_v2" ]]; then
    echo "[train_court_synthetic_v2] expected canonical_court_dataset_v2, got ${DATASET_SCHEMA}." >&2
    exit 1
fi

cd "${REPO_ROOT}"

# Colab disconnect recovery: resume from the newest TensorBoard version.
RESUME_OVERRIDES=()
shopt -s nullglob
LAST_CKPTS=("${OUTPUT_PATH}"/logs/version_*/checkpoints/last.ckpt)
shopt -u nullglob
if (( ${#LAST_CKPTS[@]} > 0 )); then
    RESUME_CKPT="$(printf '%s\n' "${LAST_CKPTS[@]}" | sort -V | tail -1)"
    case "${RESUME_CKPT}" in
        "${CHECKPOINT_ROOT%/}"/*)
            RESUME_RELATIVE="${RESUME_CKPT#"${CHECKPOINT_ROOT%/}/"}"
            ;;
        *)
            echo "[train_court_synthetic_v2] latest checkpoint is outside CHECKPOINT_ROOT: ${RESUME_CKPT}" >&2
            exit 1
            ;;
    esac
    echo "[train_court_synthetic_v2] resuming from ${RESUME_CKPT}"
    RESUME_OVERRIDES=("run.resume=${RESUME_RELATIVE}")
fi

echo "[train_court_synthetic_v2] starting training (extra overrides: ${*:-none})"
python -m src.tasks.court_detection.scripts.train \
    data/source=synthetic_court \
    data.source.schema=v2 \
    data.source.workspace_root=synthetic_data_generation/scenes \
    "data.source.scene_ids=[B00]" \
    data/processing=kp \
    "data.augmentation.train_scales=[${INPUT_SIZE}]" \
    "data.augmentation.val_short_side=${INPUT_SIZE}" \
    model/encoder=dinov3 \
    model/decoder=dpt \
    training=lora \
    "data.batch_size=${BATCH_SIZE}" \
    "training.trainer.max_epochs=${MAX_EPOCHS}" \
    training.checkpoint.monitor=val/kp_mean_dist \
    training.checkpoint.mode=min \
    training.early_stopping.monitor=val/kp_mean_dist \
    training.early_stopping.mode=min \
    run.test_after_fit=true \
    "paths.data_root=${DATA_ROOT}" \
    "paths.output_root=${OUTPUT_ROOT}" \
    "paths.checkpoint_root=${CHECKPOINT_ROOT}" \
    "run.output_dir=${OUTPUT_DIR}" \
    "${RESUME_OVERRIDES[@]}" \
    "$@"

echo "[train_court_synthetic_v2] done."
