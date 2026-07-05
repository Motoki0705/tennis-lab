#!/usr/bin/env bash
set -euo pipefail

# Common Colab setup for tennis-lab notebooks.
#
# Responsibilities:
#   - mount Google Drive at /content/drive
#   - install common system dependency: zstd
#   - install common Python dependencies used across ball/court/blcs/plcs
#
# Usage from Colab:
#   !bash scripts/colab/install_deps.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

echo "[install_deps] repo root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

echo "[install_deps] mounting Google Drive if needed..."
python - <<'PY'
from pathlib import Path

drive_root = Path("/content/drive/MyDrive")
if drive_root.exists():
    print("[install_deps] Google Drive is already mounted.")
else:
    try:
        from google.colab import drive
    except Exception as exc:
        raise SystemExit(
            "[install_deps] google.colab is unavailable. "
            "Run this script on Colab, or mount Drive manually at /content/drive."
        ) from exc

    drive.mount("/content/drive")
PY

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
