#!/usr/bin/env bash
set -euo pipefail

# Generate derived datasets for BLCS / PLCS.
#
# Supported targets:
#   blcs -> python -m src.tasks.blcs.scripts.generate_dataset
#   plcs -> python -m src.tasks.plcs.scripts.generate_dataset
#
# Usage from Colab:
#   !bash scripts/colab/prepare_generated_dataset.sh blcs
#   !bash scripts/colab/prepare_generated_dataset.sh plcs
#
# Extra Hydra overrides can be passed after the target:
#   !bash scripts/colab/prepare_generated_dataset.sh blcs run.num_workers=4
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path
#   NUM_WORKERS   default: 8
#   CACHE_DIR     default: /content/drive_upload_archives

usage() {
    cat <<'EOF'
Usage:
  bash scripts/colab/prepare_generated_dataset.sh {blcs|plcs} [hydra_overrides...]

Examples:
  bash scripts/colab/prepare_generated_dataset.sh blcs
  bash scripts/colab/prepare_generated_dataset.sh plcs
  bash scripts/colab/prepare_generated_dataset.sh blcs run.num_workers=4
EOF
}

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    usage >&2
    exit 2
fi
shift || true
TARGET="${TARGET,,}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_DIR="${CACHE_DIR:-/content/drive_upload_archives}"

case "${TARGET}" in
    blcs)
        MODULE="src.tasks.blcs.scripts.generate_dataset"
        ;;
    plcs)
        MODULE="src.tasks.plcs.scripts.generate_dataset"

        # PLCS raw archives should be prepared by prepare_archive_dataset.sh first.
        # This is intentionally a soft warning because the extracted folder names can
        # differ depending on how the tar files were created.
        missing_archives=()
        for archive in smplx.tar.zst smplh.tar.zst ACCAD.tar.zst; do
            if [[ ! -f "${CACHE_DIR}/${archive}" ]]; then
                missing_archives+=("${archive}")
            fi
        done
        if (( ${#missing_archives[@]} > 0 )); then
            echo "[prepare_generated_dataset] warning: PLCS archive cache is incomplete: ${missing_archives[*]}" >&2
            echo "[prepare_generated_dataset] recommended before this step:" >&2
            echo "  bash scripts/colab/prepare_archive_dataset.sh plcs" >&2
        fi
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "[prepare_generated_dataset] unknown target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

echo "[prepare_generated_dataset] repo root: ${REPO_ROOT}"
echo "[prepare_generated_dataset] target: ${TARGET}"
echo "[prepare_generated_dataset] module: ${MODULE}"

cd "${REPO_ROOT}"

if (( "$#" == 0 )); then
    echo "[prepare_generated_dataset] running with run.num_workers=${NUM_WORKERS}"
    python -m "${MODULE}" "run.num_workers=${NUM_WORKERS}"
else
    echo "[prepare_generated_dataset] running with overrides: $*"
    python -m "${MODULE}" "$@"
fi

echo "[prepare_generated_dataset] done."
