#!/usr/bin/env bash
set -euo pipefail

# Compare a local file or directory with its Drive copy using SHA-256.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_drive_tool verify "$@"
