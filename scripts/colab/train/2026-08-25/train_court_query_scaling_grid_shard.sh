#!/usr/bin/env bash
set -euo pipefail

# Run one deterministic half of the Issue #790 DPT scaling grid.
# The grid is input long-side {256,384,512} x task-encoder depth {1,8}
# x DPT decoder size {tiny,small,base,large}; every condition is run once.

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
TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR:-${DRIVE_QUEUE_ROOT}/issue-790/grid-${SHARD_NAME}}"
SESSION_ID="${SESSION_ID:-issue790-grid-${SHARD_NAME}}"
QUEUE_SCRIPT="${REPO_ROOT}/.agents/skills/training-queue/scripts/training_queue.sh"
DATASET_ROOT="${DATA_ROOT%/}/issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes/B00/datasets/court"
DERIVED_ROOT="${DATA_ROOT%/}/court_detection/derived_targets_issue790_v3"
BACKBONE_CKPT="${REPO_ROOT}/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

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
INITIAL_QUEUE_STATUS="$(TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" bash "${QUEUE_SCRIPT}" status)"
if [[ "${INITIAL_QUEUE_STATUS}" != *"worker: stopped"* \
      || "${INITIAL_QUEUE_STATUS}" != *"queued=0 running=0 done=0 failed=0"* ]]; then
    echo "[issue790] Grid shard requires a fresh, stopped queue directory:" >&2
    echo "${INITIAL_QUEUE_STATUS}" >&2
    exit 1
fi

declare -a RUNS=()
for INPUT_SIZE in 256 384 512; do
    for DEPTH in 1 8; do
        for DECODER_SIZE in tiny small base large; do
            RUNS+=("${INPUT_SIZE}:${DEPTH}:${DECODER_SIZE}")
        done
    done
done

if [[ "${SHARD_NAME}" == "colab-1" ]]; then
    START=0
    END=12
else
    START=12
    END=24
fi

dpt_taps() {
    local depth="$1"
    local size="$2"
    if [[ "${depth}" == 1 ]]; then
        printf '%s' '[0]'
    elif [[ "${size}" == tiny ]]; then
        printf '%s' '[0,7]'
    else
        printf '%s' '[0,2,5,7]'
    fi
}

dpt_factors() {
    local depth="$1"
    local size="$2"
    if [[ "${depth}" == 1 ]]; then
        printf '%s' '[1.0]'
    elif [[ "${size}" == tiny ]]; then
        printf '%s' '[2.0,1.0]'
    else
        printf '%s' '[4.0,2.0,1.0,0.5]'
    fi
}

for ((index=START; index<END; index++)); do
    IFS=: read -r INPUT_SIZE DEPTH DECODER_SIZE <<<"${RUNS[index]}"
    if [[ "${INPUT_SIZE}" == 256 ]]; then
        BATCH_SIZE=8
        ACCUMULATE=1
    elif [[ "${INPUT_SIZE}" == 384 ]]; then
        BATCH_SIZE=4
        ACCUMULATE=2
    else
        BATCH_SIZE=2
        ACCUMULATE=4
    fi
    TAPS="$(dpt_taps "${DEPTH}" "${DECODER_SIZE}")"
    FACTORS="$(dpt_factors "${DEPTH}" "${DECODER_SIZE}")"
    RUN_ID="input-${INPUT_SIZE}-depth-${DEPTH}-dpt-${DECODER_SIZE}"
    RELATIVE_DIR="court_detection/query_consistency_grid/${RUN_ID}"
    RUN_DIR="${OUTPUT_ROOT%/}/${RELATIVE_DIR}"
    if [[ -e "${RUN_DIR}" ]]; then
        echo "[issue790] run output must be absent before a fresh grid shard: ${RUN_DIR}" >&2
        exit 1
    fi

    TRAIN_CMD=("${RUNTIME_PYTHON}" -m src.tasks.court_detection.scripts.train
        data/source=synthetic_court
        "paths.data_root=${DATA_ROOT}"
        "paths.external_asset_root=${REPO_ROOT}/third_party"
        data.source.workspace_root=issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes
        data.source.keypoint_court_scope=target_court
        data/processing=all
        data.processing.derived_target_root=court_detection/derived_targets_issue790_v3
        data/augmentation=pose_safe
        "data.augmentation.train_scales=[${INPUT_SIZE}]"
        "data.augmentation.val_short_side=${INPUT_SIZE}"
        data.augmentation.canvas_size=null
        data.augmentation.patch_size=16
        "data.batch_size=${BATCH_SIZE}"
        model=query_encoder_base
        model.preset=raw
        model/backbone=query_dinov3
        model.backbone.train_mode=frozen
        model.backbone.last_n_blocks=0
        model.backbone.lora.enabled=false
        model/task_encoder=query_base
        "model/task_encoder.depth=${DEPTH}"
        "model/task_encoder.tap_indices=${TAPS}"
        "model/decoder=query_dpt_${DECODER_SIZE}"
        "model/decoder.tap_indices=${TAPS}"
        "model/decoder.fusion_levels=$(awk -F, '{print NF}' <<<"${TAPS#[}")"
        "model/decoder.reassemble_factors=${FACTORS}"
        model/heads=query_base
        model.heads.dense_targets=[kp,seg,line]
        loss=query_joint_both
        training.trainer.max_epochs=15
        "training.trainer.accumulate_grad_batches=${ACCUMULATE}"
        training.learning_rate=0.001
        training.weight_decay=0.0001
        training.optimizer.name=adamw
        training.optimizer.betas=[0.9,0.999]
        run.seed=42
        "run.output_dir=${RELATIVE_DIR}"
        run.test_after_fit=true
        "paths.artifact_root=outputs/${RELATIVE_DIR}/artifacts")
    printf -v TRAIN_TEXT '%q ' "${TRAIN_CMD[@]}"
    TRAIN_TEXT+=""
    TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" bash "${QUEUE_SCRIPT}" add "${TRAIN_TEXT}" \
        --name "grid-${RUN_ID}" --provider codex --session "${SESSION_ID}" --issue 790 --prune-ckpt
done

echo "[issue790] starting ${SHARD_NAME} grid jobs (${START}-${END})."
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
