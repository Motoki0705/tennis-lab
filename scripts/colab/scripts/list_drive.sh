#!/usr/bin/env bash
set -euo pipefail

# List files and directories below the mounted tennis_lab Drive root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_drive_tool list "$@"
