#!/usr/bin/env bash
set -euo pipefail

# Prepare archive-based datasets/assets from Google Drive.
#
# Supported targets:
#   ball   -> tennis.tar.zst
#   court  -> court.tar.zst
#   plcs   -> smplx.tar.zst, smplh.tar.zst, ACCAD.tar.zst
#
# Usage from Colab:
#   !bash scripts/colab/prepare_archive_dataset.sh ball
#   !bash scripts/colab/prepare_archive_dataset.sh court
#   !bash scripts/colab/prepare_archive_dataset.sh plcs
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path
#   DRIVE_DATA    default: /content/drive/MyDrive/tennis_lab/data
#   CACHE_DIR     default: /content/drive_upload_archives
#   DATA_DIR      default: ${REPO_ROOT}/data

usage() {
    cat <<'EOF'
Usage:
  bash scripts/colab/prepare_archive_dataset.sh {ball|court|plcs}

Targets:
  ball   Copy/extract tennis.tar.zst
  court  Copy/extract court.tar.zst
  plcs   Copy/extract smplx.tar.zst, smplh.tar.zst, ACCAD.tar.zst

Environment overrides:
  DRIVE_DATA=/content/drive/MyDrive/tennis_lab/data
  CACHE_DIR=/content/drive_upload_archives
  DATA_DIR=/content/tennis-lab/data
EOF
}

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    usage >&2
    exit 2
fi
TARGET="${TARGET,,}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DRIVE_DATA="${DRIVE_DATA:-/content/drive/MyDrive/tennis_lab/data}"
CACHE_DIR="${CACHE_DIR:-/content/drive_upload_archives}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

case "${TARGET}" in
    ball)
        ARCHIVES=("tennis.tar.zst")
        ;;
    court)
        ARCHIVES=("court.tar.zst")
        ;;
    plcs)
        ARCHIVES=("smplx.tar.zst" "smplh.tar.zst" "ACCAD.tar.zst")
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "[prepare_archive_dataset] unknown target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

if [[ ! -d /content/drive/MyDrive ]]; then
    echo "[prepare_archive_dataset] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[prepare_archive_dataset] Run scripts/colab/install_deps.sh first." >&2
    exit 1
fi

if ! command -v zstd >/dev/null 2>&1; then
    echo "[prepare_archive_dataset] zstd is not installed." >&2
    echo "[prepare_archive_dataset] Run scripts/colab/install_deps.sh first." >&2
    exit 1
fi

echo "[prepare_archive_dataset] target: ${TARGET}"
echo "[prepare_archive_dataset] drive data: ${DRIVE_DATA}"
echo "[prepare_archive_dataset] cache dir: ${CACHE_DIR}"
echo "[prepare_archive_dataset] data dir: ${DATA_DIR}"

mkdir -p "${CACHE_DIR}" "${DATA_DIR}"

for archive in "${ARCHIVES[@]}"; do
    src="${DRIVE_DATA}/${archive}"
    dst="${CACHE_DIR}/${archive}"

    if [[ ! -f "${src}" ]]; then
        echo "[prepare_archive_dataset] missing source archive: ${src}" >&2
        exit 1
    fi

    if [[ ! -f "${dst}" || "${src}" -nt "${dst}" ]]; then
        echo "[prepare_archive_dataset] copying ${archive}..."
        cp -av "${src}" "${dst}"
    else
        echo "[prepare_archive_dataset] cache already exists: ${dst}"
    fi
done

for archive in "${ARCHIVES[@]}"; do
    echo "[prepare_archive_dataset] extracting ${archive} to ${DATA_DIR}..."
    tar -I zstd -xf "${CACHE_DIR}/${archive}" -C "${DATA_DIR}"
done

echo "[prepare_archive_dataset] done."
