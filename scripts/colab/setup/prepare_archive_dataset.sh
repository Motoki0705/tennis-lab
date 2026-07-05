#!/usr/bin/env bash
set -euo pipefail

# Prepare archive-based datasets/assets from Google Drive.
#
# Supported targets:
#   ball   -> tennis.tar.zst
#   court  -> court.tar.zst, DINOv3 submodule, DINOv3 ViT-B/16 checkpoint
#   plcs   -> smplx.tar.zst, smplh.tar.zst, ACCAD.tar.zst
#
# Usage from Colab:
#   !bash scripts/colab/setup/prepare_archive_dataset.sh ball
#   !bash scripts/colab/setup/prepare_archive_dataset.sh court
#   !bash scripts/colab/setup/prepare_archive_dataset.sh plcs
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path
#   DRIVE_DATA    default: /content/drive/MyDrive/tennis_lab/data
#   CACHE_DIR     default: /content/drive_upload_archives
#   DATA_DIR      default: ${REPO_ROOT}/data
#   DINOV3_CKPT   default: ${DRIVE_DATA}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

usage() {
    cat <<'EOF'
Usage:
  bash scripts/colab/setup/prepare_archive_dataset.sh {ball|court|plcs}

Targets:
  ball   Copy/extract tennis.tar.zst
  court  Copy/extract court.tar.zst and stage DINOv3 assets
  plcs   Copy/extract smplx.tar.zst, smplh.tar.zst, ACCAD.tar.zst

Environment overrides:
  DRIVE_DATA=/content/drive/MyDrive/tennis_lab/data
  CACHE_DIR=/content/drive_upload_archives
  DATA_DIR=/content/tennis-lab/data
  DINOV3_CKPT=/content/drive/MyDrive/tennis_lab/data/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
EOF
}

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
    usage >&2
    exit 2
fi
TARGET="${TARGET,,}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DRIVE_DATA="${DRIVE_DATA:-/content/drive/MyDrive/tennis_lab/data}"
CACHE_DIR="${CACHE_DIR:-/content/drive_upload_archives}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
DINOV3_CKPT_NAME="dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DINOV3_CKPT="${DINOV3_CKPT:-${DRIVE_DATA}/${DINOV3_CKPT_NAME}}"
DINOV3_DEST_DIR="${REPO_ROOT}/third_party/dinov3/checkpoints"
DINOV3_DEST="${DINOV3_DEST_DIR}/${DINOV3_CKPT_NAME}"

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

prepare_dinov3_assets() {
    echo "[prepare_archive_dataset] preparing DINOv3 submodule..."
    if [[ ! -d "${REPO_ROOT}/third_party/dinov3/dinov3" ]]; then
        if ! git -C "${REPO_ROOT}" submodule update --init third_party/dinov3; then
            echo "[prepare_archive_dataset] SSH submodule update failed; retrying DINOv3 over HTTPS..."
            git -C "${REPO_ROOT}" config submodule.third_party/dinov3.url \
                https://github.com/Motoki0705/dinov3.git
            git -C "${REPO_ROOT}" submodule sync third_party/dinov3
            git -C "${REPO_ROOT}" submodule update --init third_party/dinov3
        fi
    else
        echo "[prepare_archive_dataset] DINOv3 submodule already initialized."
    fi

    if [[ ! -d "${REPO_ROOT}/third_party/dinov3" ]]; then
        echo "[prepare_archive_dataset] DINOv3 submodule directory is missing." >&2
        exit 1
    fi

    if [[ ! -f "${DINOV3_CKPT}" ]]; then
        echo "[prepare_archive_dataset] missing DINOv3 checkpoint: ${DINOV3_CKPT}" >&2
        echo "[prepare_archive_dataset] Expected the checkpoint in Google Drive, or set DINOV3_CKPT." >&2
        exit 1
    fi

    mkdir -p "${DINOV3_DEST_DIR}"
    if [[ ! -f "${DINOV3_DEST}" || "${DINOV3_CKPT}" -nt "${DINOV3_DEST}" ]]; then
        echo "[prepare_archive_dataset] copying DINOv3 checkpoint..."
        cp -av "${DINOV3_CKPT}" "${DINOV3_DEST}"
    else
        echo "[prepare_archive_dataset] DINOv3 checkpoint already exists: ${DINOV3_DEST}"
    fi
}

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

if [[ "${TARGET}" == "court" ]]; then
    prepare_dinov3_assets
fi

echo "[prepare_archive_dataset] done."
