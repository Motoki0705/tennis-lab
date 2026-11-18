#!/usr/bin/env bash
# The measured command is preserved in run.json.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "$SCRIPT_DIR/../run-person-stage-precision-meiji-20261118/reproduce.sh" smoke 5854fa629ac8891ae9210d553c72e3c9ca5a9213 "$@"
