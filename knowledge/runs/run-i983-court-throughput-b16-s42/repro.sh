#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Execute on a managed Colab L4 at c4b12f923 after the shared input verification.
# Submit with --persist matching the fresh output directory passed below.
PROFILE_OUTPUT="${1:?Provide a fresh output directory}"
PYTHONPATH=. .venv/bin/python "$SCRIPT_DIR/profile_throughput.py" \
  --output "$PROFILE_OUTPUT" --batch-size 16
