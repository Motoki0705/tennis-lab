#!/usr/bin/env bash
set -euo pipefail

# Common Colab setup for tennis-lab notebooks.
#
# Responsibilities:
#   - verify Google Drive is already mounted at /content/drive
#   - install common system dependency: zstd
#   - install common Python dependencies used across ball/court/blcs/plcs
#
# Usage from Colab:
#   from google.colab import drive
#   drive.mount("/content/drive")
#   !bash scripts/colab/setup/install_deps.sh
#
# Environment overrides:
#   REPO_ROOT     default: repository root inferred from this script path

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

echo "[install_deps] repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

echo "[install_deps] checking Google Drive mount..."
if [[ ! -d /content/drive/MyDrive ]]; then
    echo "[install_deps] Google Drive is not mounted at /content/drive/MyDrive." >&2
    echo "[install_deps] Run this in a Colab Python cell before this script:" >&2
    echo "from google.colab import drive" >&2
    echo "drive.mount('/content/drive')" >&2
    exit 1
fi

echo "[install_deps] installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y zstd
else
    echo "[install_deps] apt-get not found. Skipping apt dependencies."
fi

echo "[install_deps] installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install hydra-core pytorch-lightning smplx

echo "[install_deps] done."
