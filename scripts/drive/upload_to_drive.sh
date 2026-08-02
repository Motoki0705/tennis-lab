#!/usr/bin/env bash
set -euo pipefail

# Upload one local file or directory to an exact Drive-relative destination.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_drive_tool upload "$@"
