#!/usr/bin/env bash
set -euo pipefail

# Search names below the configured rclone Drive root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_drive_tool search "$@"
