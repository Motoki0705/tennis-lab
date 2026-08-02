#!/usr/bin/env bash
set -euo pipefail

# Inspect metadata and optional checksums for one Drive path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
run_drive_tool inspect "$@"
