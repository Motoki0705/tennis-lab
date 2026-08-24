#!/usr/bin/env bash
set -euo pipefail

# Internal runner for one Issue #790 encoder-scaling seed shard.
# Invoke one of the colab0/colab1 wrappers instead of calling this directly.

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 {colab-0|colab-1} {43|44}" >&2
    exit 2
fi

SHARD_NAME="$1"
SEED="$2"
case "${SHARD_NAME}:${SEED}" in
    colab-0:43|colab-1:44) ;;
    *)
        echo "[issue790] shard/seed assignment must be colab-0:43 or colab-1:44." >&2
        exit 2
        ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/tennis_lab/outputs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${OUTPUT_ROOT}}"
DRIVE_QUEUE_ROOT="${DRIVE_QUEUE_ROOT:-/content/drive/MyDrive/tennis_lab/training_queue}"
TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR:-${DRIVE_QUEUE_ROOT}/issue-790/${SHARD_NAME}}"
SESSION_ID="${SESSION_ID:-issue790-${SHARD_NAME}-seed-${SEED}}"
MANIFEST_PATH="${REPO_ROOT}/outputs/court_detection/query_consistency_ablation/manifest.json"
QUEUE_SCRIPT="${REPO_ROOT}/.agents/skills/training-queue/scripts/training_queue.sh"
DATASET_ROOT="${DATA_ROOT%/}/issue-779-court-query-v3-attempt9/synthetic_data_generation/scenes/B00/datasets/court"
DERIVED_ROOT="${DATA_ROOT%/}/court_detection/derived_targets_issue790_v3"
BACKBONE_CKPT="${REPO_ROOT}/third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# shellcheck source=scripts/colab/setup/install_deps.sh
source "${REPO_ROOT}/scripts/colab/setup/install_deps.sh"
# shellcheck source=scripts/colab/setup/prepare_archive_dataset.sh
source "${REPO_ROOT}/scripts/colab/setup/prepare_archive_dataset.sh"
# shellcheck source=scripts/colab/setup/path_contract.sh
source "${REPO_ROOT}/scripts/colab/setup/path_contract.sh"

validate_colab_role_root DATA_ROOT "${DATA_ROOT}"
validate_colab_role_root OUTPUT_ROOT "${OUTPUT_ROOT}"
validate_colab_role_root CHECKPOINT_ROOT "${CHECKPOINT_ROOT}"
validate_colab_role_root TRAINING_QUEUE_DIR "${TRAINING_QUEUE_DIR}"

if [[ ! -d /content/drive/MyDrive ]]; then
    echo "[issue790] Google Drive is not mounted at /content/drive/MyDrive." >&2
    exit 1
fi

echo "[issue790] shard: ${SHARD_NAME}"
echo "[issue790] seed: ${SEED}"
echo "[issue790] repo root: ${REPO_ROOT}"
echo "[issue790] output root: ${OUTPUT_ROOT}"
echo "[issue790] queue: ${TRAINING_QUEUE_DIR}"

install_colab_dependencies "${REPO_ROOT}"
DATA_DIR="${DATA_ROOT}" prepare_archive_dataset court_query_issue790_v3 "${REPO_ROOT}"

if [[ ! -f "${DATASET_ROOT}/dataset.json" ]]; then
    echo "[issue790] V3 dataset is missing after archive staging: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${DERIVED_ROOT}" ]]; then
    echo "[issue790] V3 derived targets are missing after archive staging: ${DERIVED_ROOT}" >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" || ! -f "${BACKBONE_CKPT}" ]]; then
    echo "[issue790] DINOv3 source/checkpoint is missing after archive staging." >&2
    exit 1
fi

python -c '
import json
import pathlib
import sys

dataset_path = pathlib.Path(sys.argv[1])
derived_root = pathlib.Path(sys.argv[2])
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
schema_key = "schema"
if dataset.get(schema_key) != "canonical_court_dataset_v3":
    raise SystemExit(f"unexpected dataset schema: {dataset.get(schema_key)!r}")
if dataset.get("status") != "completed" or len(dataset.get("samples", ())) != 3293:
    raise SystemExit("Issue #790 requires the completed 3,293-sample V3 dataset")
for schema in ("court_cell_segmentation_v1", "court_line_binary_v1"):
    root = derived_root / "synthetic_court" / schema / "B00"
    if len(tuple(root.glob("*.png"))) != 3293 or len(tuple(root.glob("*.json"))) != 3293:
        raise SystemExit(f"Issue #790 derived-target inventory is incomplete: {root}")
' "${DATASET_ROOT}/dataset.json" "${DERIVED_ROOT}"

cd "${REPO_ROOT}"
python -m src.tasks.court_detection.scripts.run_query_consistency_ablation

RUNTIME_PYTHON="$(command -v python)"
INITIAL_QUEUE_STATUS="$(TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" bash "${QUEUE_SCRIPT}" status)"
if [[ "${INITIAL_QUEUE_STATUS}" != *"worker: stopped"* \
      || "${INITIAL_QUEUE_STATUS}" != *"queued=0 running=0 done=0 failed=0"* ]]; then
    echo "[issue790] Colab shard requires a fresh, stopped queue directory:" >&2
    echo "${INITIAL_QUEUE_STATUS}" >&2
    exit 1
fi
for DEPTH in 01 02 04 08; do
    RUN_DIR="${OUTPUT_ROOT%/}/court_detection/query_consistency_ablation/encoder-depth-${DEPTH}-seed-${SEED}"
    if [[ -e "${RUN_DIR}" ]]; then
        echo "[issue790] run output must be absent before a fresh shard: ${RUN_DIR}" >&2
        exit 1
    fi
done
python -m scripts.colab.setup.enqueue_query_consistency_shard \
    --manifest "${MANIFEST_PATH}" \
    --seed "${SEED}" \
    --job-kind both \
    --python-executable "${RUNTIME_PYTHON}" \
    --data-root "${DATA_ROOT}" \
    --external-asset-root "${REPO_ROOT}/third_party" \
    --output-root "${OUTPUT_ROOT}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --queue-script "${QUEUE_SCRIPT}" \
    --queue-dir "${TRAINING_QUEUE_DIR}" \
    --repository-root "${REPO_ROOT}" \
    --provider codex \
    --session "${SESSION_ID}" \
    --issue 790

echo "[issue790] starting four training and four profile jobs in the foreground."
TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" \
TRAINING_QUEUE_PYTHON="${RUNTIME_PYTHON}" \
bash "${QUEUE_SCRIPT}" serve --idle-timeout 30

FINAL_QUEUE_STATUS="$(TRAINING_QUEUE_DIR="${TRAINING_QUEUE_DIR}" bash "${QUEUE_SCRIPT}" status)"
echo "${FINAL_QUEUE_STATUS}"
if [[ "${FINAL_QUEUE_STATUS}" != *"worker: stopped"* \
      || "${FINAL_QUEUE_STATUS}" != *"queued=0 running=0 done=8 failed=0"* ]]; then
    echo "[issue790] one or more shard jobs failed; inspect ${TRAINING_QUEUE_DIR}/logs." >&2
    exit 1
fi
echo "[issue790] ${SHARD_NAME} seed ${SEED} shard finished."
