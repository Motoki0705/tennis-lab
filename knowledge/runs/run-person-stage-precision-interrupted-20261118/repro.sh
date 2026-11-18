#!/usr/bin/env bash
# The measured command is preserved in run.json.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/../run-person-stage-precision-meiji-20261118/reproduce.sh" full ba5d26a0767955563ed24a85bb0d9d8bddcc6c16 "$@"
