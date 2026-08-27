#!/usr/bin/env bash
# Enqueue the refreshed Issue #790 Court Detection condition/scaling matrix.
#
# This entry point freezes a deterministic manifest and only admits its jobs to
# training_queue.sh. It never starts a queue worker; the shared supervisor owns
# GPU serialization.
#
# Usage:
#   TRAINING_QUEUE_PROVIDER=codex \
#   TRAINING_QUEUE_SESSION=<session-id> \
#   bash scripts/training/enqueue_court_detection_matrix.sh
#
# Optional environment overrides:
#   TRAINING_QUEUE_DIR       queue state (defaults to the main checkout queue)
#   TRAINING_QUEUE_ISSUE     must be 790 when set (defaults to 790)
#   TRAINING_PYTHON          Python executable (defaults to .venv/bin/python)
#   COURT_MAX_EPOCHS         fixed training budget (must be 15)
#   COURT_BATCH_SIZE         fixed batch size (must be 8)
#   COURT_SEED               fixed initial matrix seed (must be 42)
#   COURT_RUN_PREFIX         output prefix (defaults to court_detection/matrix)
#   COURT_MANIFEST_PATH      frozen manifest output path
#   COURT_CONSISTENCY_WEIGHT pure consistency weight (must be 1.0)
#   COURT_WEIGHTED_KP        weighted job KP weight (defaults to 1.0)
#   COURT_WEIGHTED_LINE      weighted job LINE weight (defaults to 0.5)
#   COURT_WEIGHTED_SEG       weighted job SEG weight (defaults to 0.5)
#   COURT_WEIGHTED_POSE_T    weighted translation weight (defaults to 0.01)
#   COURT_WEIGHTED_POSE_R    weighted rotation weight (defaults to 0.5)
#   COURT_WEIGHTED_POSE_F    weighted focal weight (defaults to 0.05)
#   COURT_WEIGHTED_AUX       weighted consistency weight (defaults to 0.25)
#
# The 21 unique jobs contain six primary conditions (KP-only, LINE-only,
# SEG-only, pose-only, pure, weighted) and every pure depth 1/8 x input
# 256/384 x DPT tiny/small/base/large combination. The pure d8/i256/large job
# serves both condition and scaling roles, avoiding a duplicate GPU run.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)}"
readonly QUEUE_SCRIPT="${QUEUE_SCRIPT:-${REPO_ROOT}/.agents/skills/training-queue/scripts/training_queue.sh}"
readonly MATRIX_TOOL="${SCRIPT_DIR}/court_detection_matrix.py"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '1,39p' "${BASH_SOURCE[0]}"
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: TRAINING_QUEUE_PROVIDER=... TRAINING_QUEUE_SESSION=... bash $0" >&2
    exit 2
fi

: "${TRAINING_QUEUE_PROVIDER:?TRAINING_QUEUE_PROVIDER is required for queue attribution}"
: "${TRAINING_QUEUE_SESSION:?TRAINING_QUEUE_SESSION is required for queue attribution}"

readonly PROVIDER="${TRAINING_QUEUE_PROVIDER}"
readonly SESSION="${TRAINING_QUEUE_SESSION}"
readonly ISSUE="${TRAINING_QUEUE_ISSUE:-790}"
readonly PYTHON="${TRAINING_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
readonly MAX_EPOCHS="${COURT_MAX_EPOCHS:-15}"
readonly BATCH_SIZE="${COURT_BATCH_SIZE:-8}"
readonly SEED="${COURT_SEED:-42}"
readonly RUN_PREFIX="${COURT_RUN_PREFIX:-court_detection/matrix}"
readonly CONSISTENCY_WEIGHT="${COURT_CONSISTENCY_WEIGHT:-1.0}"
readonly WEIGHTED_KP="${COURT_WEIGHTED_KP:-1.0}"
readonly WEIGHTED_LINE="${COURT_WEIGHTED_LINE:-0.5}"
readonly WEIGHTED_SEG="${COURT_WEIGHTED_SEG:-0.5}"
readonly WEIGHTED_POSE_T="${COURT_WEIGHTED_POSE_T:-0.01}"
readonly WEIGHTED_POSE_R="${COURT_WEIGHTED_POSE_R:-0.5}"
readonly WEIGHTED_POSE_F="${COURT_WEIGHTED_POSE_F:-0.05}"
readonly WEIGHTED_AUX="${COURT_WEIGHTED_AUX:-0.25}"

if [[ -z "${REPO_ROOT}" || ! -d "${REPO_ROOT}" ]]; then
    echo "Unable to resolve the repository root." >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Training Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${MATRIX_TOOL}" ]]; then
    echo "Court matrix tool is missing: ${MATRIX_TOOL}" >&2
    exit 1
fi
if [[ ! -x "${QUEUE_SCRIPT}" ]]; then
    echo "Training queue script is not executable: ${QUEUE_SCRIPT}" >&2
    exit 1
fi
if [[ "${ISSUE}" != "790" ]]; then
    echo "TRAINING_QUEUE_ISSUE must be 790 for this frozen matrix: ${ISSUE}" >&2
    exit 1
fi

readonly GIT_COMMON_DIR="$(git -C "${REPO_ROOT}" rev-parse --git-common-dir)"
if [[ "${GIT_COMMON_DIR}" = /* ]]; then
    readonly MAIN_REPO_ROOT="$(cd -- "$(dirname -- "${GIT_COMMON_DIR}")" && pwd)"
else
    readonly MAIN_REPO_ROOT="$(cd -- "${REPO_ROOT}/$(dirname -- "${GIT_COMMON_DIR}")" && pwd)"
fi
QUEUE_DIR="${TRAINING_QUEUE_DIR:-${MAIN_REPO_ROOT}/.training_queue}"
if [[ "${QUEUE_DIR}" != /* ]]; then
    QUEUE_DIR="${REPO_ROOT}/${QUEUE_DIR}"
fi
MANIFEST_PATH="${COURT_MANIFEST_PATH:-${REPO_ROOT}/outputs/${RUN_PREFIX}/manifest.json}"
if [[ "${MANIFEST_PATH}" != /* ]]; then
    MANIFEST_PATH="${REPO_ROOT}/${MANIFEST_PATH}"
fi

matrix_args=(
    --python "${PYTHON}"
    --max-epochs "${MAX_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --seed "${SEED}"
    --run-prefix "${RUN_PREFIX}"
    --manifest-path "${MANIFEST_PATH}"
    --consistency-weight "${CONSISTENCY_WEIGHT}"
    --weighted-kp "${WEIGHTED_KP}"
    --weighted-line "${WEIGHTED_LINE}"
    --weighted-seg "${WEIGHTED_SEG}"
    --weighted-pose-translation "${WEIGHTED_POSE_T}"
    --weighted-pose-rotation "${WEIGHTED_POSE_R}"
    --weighted-pose-focal "${WEIGHTED_POSE_F}"
    --weighted-consistency "${WEIGHTED_AUX}"
)

jobs_file="$(mktemp)"
trap 'rm -f -- "${jobs_file}"' EXIT
"${PYTHON}" "${MATRIX_TOOL}" validate-configs "${matrix_args[@]}" > /dev/null
"${PYTHON}" "${MATRIX_TOOL}" emit-jobs "${matrix_args[@]}" > "${jobs_file}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON}" "${MATRIX_TOOL}" generate-manifest \
        "${matrix_args[@]}" \
        --output "${MANIFEST_PATH}"
fi

job_count=0
while IFS=$'\t' read -r queue_name command; do
    if [[ -z "${queue_name}" || -z "${command}" ]]; then
        echo "Matrix tool emitted an invalid empty queue record." >&2
        exit 1
    fi
    queue_args=(
        add "${command}"
        --name "${queue_name}"
        --provider "${PROVIDER}"
        --session "${SESSION}"
        --issue "${ISSUE}"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'dry-run: %s\n  %s\n' "${queue_name}" "${command}"
    else
        (
            cd -- "${REPO_ROOT}"
            TRAINING_QUEUE_DIR="${QUEUE_DIR}" bash "${QUEUE_SCRIPT}" "${queue_args[@]}"
        )
    fi
    job_count=$((job_count + 1))
done < "${jobs_file}"

if [[ "${job_count}" -ne 21 ]]; then
    echo "Expected 21 unique matrix jobs, received ${job_count}." >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "No jobs or manifest were written (DRY_RUN=1); validated ${job_count} jobs."
else
    echo "${job_count} Court Detection jobs admitted; queue worker was not started."
    echo "Frozen manifest: ${MANIFEST_PATH}"
    echo "Queue state: ${QUEUE_DIR}"
    echo "Start/observe separately: ${QUEUE_SCRIPT} start | ${QUEUE_SCRIPT} status"
fi
