#!/usr/bin/env bash
# CPU replay of the delivered files. Download the pinned inputs, manifests,
# 8 audits and 14 MP4s to the paths below first; no GPU is used here.
set -euo pipefail
REPO="${TENNIS_REPO:?set the pinned tennis-lab checkout}"
cd "$REPO"
test "$(git rev-parse HEAD)" = 1231c2698369f0853368fe90faa3a4439eaa7925
# --- original training command ---
.venv/bin/python "$REPO/knowledge/runs/run-player-camera-stabilization-20261109/verify_delivery.py" --inputs outputs/tennis_scene/visualize/player_view/stabilization-comparison/inputs --run outputs/tennis_scene/visualize/player_view/stabilization-comparison/attempt2/full --output outputs/tennis_scene/visualize/player_view/stabilization-comparison/operations/attempt2/delivery-verification.json
