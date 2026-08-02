#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the Colab Drive utility entry points.

COLAB_DRIVE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLAB_DRIVE_ROOT="${TENNIS_LAB_DRIVE_ROOT:-/content/drive/MyDrive/tennis_lab}"
COLAB_PYTHON="${COLAB_PYTHON:-python3}"

run_drive_tool() {
    local command_name="$1"
    shift

    if ! command -v "${COLAB_PYTHON}" >/dev/null 2>&1; then
        echo "[colab-drive-tools] Python executable not found: ${COLAB_PYTHON}" >&2
        exit 1
    fi

    exec "${COLAB_PYTHON}" \
        "${COLAB_DRIVE_SCRIPTS_DIR}/lib/drive_tools.py" \
        --drive-root "${COLAB_DRIVE_ROOT}" \
        "${command_name}" "$@"
}
