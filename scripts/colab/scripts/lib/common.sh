#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the local Google Drive utility entry points.

DRIVE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRIVE_REMOTE_ROOT="${TENNIS_LAB_DRIVE_REMOTE:-gdrive:tennis_lab}"
DRIVE_TOOLS_PYTHON="${DRIVE_TOOLS_PYTHON:-python3}"
RCLONE_BIN="${RCLONE_BIN:-rclone}"

run_drive_tool() {
    local command_name="$1"
    shift

    if ! command -v "${DRIVE_TOOLS_PYTHON}" >/dev/null 2>&1; then
        echo "[drive-tools] Python executable not found: ${DRIVE_TOOLS_PYTHON}" >&2
        exit 1
    fi

    exec "${DRIVE_TOOLS_PYTHON}" \
        "${DRIVE_SCRIPTS_DIR}/lib/drive_tools.py" \
        --remote-root "${DRIVE_REMOTE_ROOT}" \
        --rclone-bin "${RCLONE_BIN}" \
        "${command_name}" "$@"
}
