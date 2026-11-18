#!/usr/bin/env bash
# The measured command is preserved in run.json.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/reproduce.sh" full cbefdef4bc913fadc574be83af38e83f8c3f931d "$@"
