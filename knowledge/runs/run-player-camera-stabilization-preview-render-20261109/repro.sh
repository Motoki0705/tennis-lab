#!/usr/bin/env bash
# Replay the archived helper; colab-original-repro.sh preserves the historical command.
# Prepare the owned Colab VM, pinned inputs, queue receipts and a fresh explicit
# gpu-operation deadline using the skills and reproducibility-note.md first.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:?set TENNIS_REPO to a tennis-lab checkout}"
: "${TENNIS_REPRO_PREPARED_OWNED_VM:?prepare the owned VM and fresh explicit budget first}"
cd "$REPO"
test "$(git rev-parse HEAD)" = d133993421c206af28340f4ffe082c65b70fe7a7 || { echo "Prepare a separate checkout at the pinned ref first" >&2; exit 2; }
OPS=outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2
# Refuse to replace a helper from an ongoing attempt.
mkdir -p "$OPS"
if [ -e "$OPS/worker_runtime.py" ]; then
  cmp "$SCRIPT_DIR/worker_runtime.py" "$OPS/worker_runtime.py"
else
  cp "$SCRIPT_DIR/worker_runtime.py" "$OPS/worker_runtime.py"
fi
# --- original training command ---
.venv/bin/python "$SCRIPT_DIR/worker_runtime.py" preview
