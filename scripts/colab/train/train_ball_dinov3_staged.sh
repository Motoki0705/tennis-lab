#!/usr/bin/env bash
set -euo pipefail

# Train the DINOv3-RoPE ball detector on Colab: staged phases 1 -> 2 -> 3
# (issue #579 schedule) with a LoRA backbone, switchable between the original
# and the tennis-SSL DINOv3 ViT-B/16 weights to measure how much the extra SSL
# changes downstream accuracy.
#
#   phase1  TrackNet only,      T=1,       10 epochs (fresh start)
#   phase2  TrackNet + Web mix, T=1,       15 epochs (from phase1 best)
#   phase3  TrackNet only,      T in [1,8], 10 epochs (from phase2 best)
#
# Issue #618 recipe is applied on top of the phase configs:
#   - checkpoints are selected by val/f1 (max), not val/loss (the two are
#     decoupled; see knowledge/nodes/run-i618-convnext-v2-ft.md)
#   - phases chain from the previous phase's *best-f1* checkpoint, not last
#     (save_top_k=1 makes the single staged-*.ckpt the best one)
#   - sub-pixel metric refinement is already the repo default (metrics config)
#
# Responsibilities:
#   - verify DINOv3 assets and the ball datasets are staged
#     (run scripts/colab/setup/prepare_archive_dataset.sh ball first)
#   - run the three phases sequentially, checkpoints/logs on Google Drive
#   - auto-resume: if a phase's last.ckpt exists on Drive, the phase resumes
#     full-state after a Colab disconnect (finished phases no-op quickly)
#
# Usage from Colab (after scripts/colab/setup/install_deps.sh and
# scripts/colab/setup/prepare_archive_dataset.sh ball):
#   !bash scripts/colab/train/train_ball_dinov3_staged.sh orig
#   !bash scripts/colab/train/train_ball_dinov3_staged.sh ssl
#
# Extra Hydra overrides are appended to every phase, e.g.:
#   !bash scripts/colab/train/train_ball_dinov3_staged.sh ssl data.num_workers=4
#
# Environment overrides:
#   REPO_ROOT      default: repository root inferred from this script path
#   OUTPUT_ROOT    default: /content/drive/MyDrive/tennis_lab/outputs/ball_detection/dinov3_staged_${VARIANT}
#   PHASE1_EPOCHS  default: 10
#   PHASE2_EPOCHS  default: 15
#   PHASE3_EPOCHS  default: 10
#   NUM_WORKERS    default: 8

usage() {
    echo "Usage: bash scripts/colab/train/train_ball_dinov3_staged.sh {orig|ssl} [hydra overrides...]" >&2
}

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
    usage
    exit 2
fi
shift

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs/ball_detection/dinov3_staged_${VARIANT}}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-10}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-15}"
PHASE3_EPOCHS="${PHASE3_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CKPT_DIR="${REPO_ROOT}/third_party/dinov3/checkpoints"
case "${VARIANT}" in
    orig)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ;;
    ssl)
        BACKBONE_CKPT="${CKPT_DIR}/dinov3_vitb16_tennis_ssl_merged.pth"
        ;;
    *)
        echo "[train_ball_dinov3_staged] unknown variant: ${VARIANT}" >&2
        usage
        exit 2
        ;;
esac

echo "[train_ball_dinov3_staged] repo root: ${REPO_ROOT}"
echo "[train_ball_dinov3_staged] variant: ${VARIANT}"
echo "[train_ball_dinov3_staged] backbone: ${BACKBONE_CKPT}"
echo "[train_ball_dinov3_staged] output root: ${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

if [[ "${OUTPUT_ROOT}" == /content/drive/* && ! -d /content/drive/MyDrive ]]; then
    echo "[train_ball_dinov3_staged] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[train_ball_dinov3_staged] Mount it first (checkpoints must survive Colab VM resets):" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" \
      || ! -d "${REPO_ROOT}/data/tennis/tracknet" || ! -d "${REPO_ROOT}/data/tennis/web/unified" ]]; then
    echo "[train_ball_dinov3_staged] missing DINOv3 assets or ball datasets" >&2
    echo "[train_ball_dinov3_staged] (need third_party/dinov3, ${BACKBONE_CKPT}," >&2
    echo "[train_ball_dinov3_staged]  data/tennis/tracknet and data/tennis/web/unified)." >&2
    echo "[train_ball_dinov3_staged] Run this first:" >&2
    echo "  bash scripts/colab/setup/prepare_archive_dataset.sh ball" >&2
    exit 1
fi

# Print the best-f1 checkpoint of a finished phase. The phase configs keep
# save_top_k=1, so exactly one staged-*.ckpt must exist next to last.ckpt.
best_ckpt() {
    local phase_dir="$1"
    local matches=()
    local path
    for path in "${phase_dir}"/logs/run/checkpoints/staged-*.ckpt; do
        [[ -f "${path}" ]] && matches+=("${path}")
    done
    if (( ${#matches[@]} != 1 )); then
        echo "[train_ball_dinov3_staged] expected exactly one best checkpoint in ${phase_dir}/logs/run/checkpoints/, found ${#matches[@]}." >&2
        exit 1
    fi
    echo "${matches[0]}"
}

# run_phase <phase> <epochs> <init_ckpt|""> [extra overrides...]
run_phase() {
    local phase="$1" epochs="$2" init_ckpt="$3"
    shift 3
    local out="${OUTPUT_ROOT}/phase${phase}"
    local last="${out}/logs/run/checkpoints/last.ckpt"

    local overrides=(
        --config-name "staged_phase${phase}"
        "model.backbone.lora.enabled=true"
        "model.backbone.checkpoint_path=${BACKBONE_CKPT}"
        "training.trainer.max_epochs=${epochs}"
        "training.checkpoint.monitor=val/f1"
        "training.checkpoint.mode=max"
        "data.num_workers=${NUM_WORKERS}"
        "run.output_dir=${out}"
    )
    if [[ -f "${last}" ]]; then
        # Colab disconnect recovery: full-state resume. run.resume and
        # run.init_weights are mutually exclusive, so null the latter.
        echo "[train_ball_dinov3_staged] phase${phase}: resuming from ${last}"
        overrides+=("run.resume=${last}" "run.init_weights=null")
    elif [[ -n "${init_ckpt}" ]]; then
        echo "[train_ball_dinov3_staged] phase${phase}: init weights from ${init_ckpt}"
        overrides+=("run.init_weights=${init_ckpt}")
    else
        overrides+=("run.init_weights=null")
    fi

    echo "[train_ball_dinov3_staged] phase${phase}: ${epochs} epochs -> ${out}"
    python -m src.tasks.ball_detection.scripts.train_staged "${overrides[@]}" "$@"
}

run_phase 1 "${PHASE1_EPOCHS}" "" "$@"
# Standalone assignments so a failed best_ckpt aborts under `set -e` instead
# of silently passing an empty init checkpoint to the next phase.
PHASE1_BEST="$(best_ckpt "${OUTPUT_ROOT}/phase1")"
run_phase 2 "${PHASE2_EPOCHS}" "${PHASE1_BEST}" "$@"
PHASE2_BEST="$(best_ckpt "${OUTPUT_ROOT}/phase2")"
run_phase 3 "${PHASE3_EPOCHS}" "${PHASE2_BEST}" "$@"
PHASE3_BEST="$(best_ckpt "${OUTPUT_ROOT}/phase3")"

echo "[train_ball_dinov3_staged] done. Best final checkpoint: ${PHASE3_BEST}"
