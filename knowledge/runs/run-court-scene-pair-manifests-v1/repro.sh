#!/usr/bin/env bash
set -euo pipefail

# Run from the active checkout. Use a fresh, untracked work directory.
PAIR_WORK_ROOT="${PAIR_WORK_ROOT:?Set an absolute, untracked work directory}"
RCLONE_CONFIG="${RCLONE_CONFIG:?Set the authenticated tennis-drive configuration path}"
PAIR_BUNDLE="knowledge/runs/run-court-scene-pair-manifests-v1"
PAIR_PYTHON=".venv/bin/python"
PAIR_DRIVE=".agents/skills/tennis-drive/scripts/drive.py"
mkdir -p "$PAIR_WORK_ROOT/receipts"

"$PAIR_PYTHON" "$PAIR_DRIVE" --rclone-config "$RCLONE_CONFIG" \
  inventory --path data/court_scene_diversity_6000_v1 \
  --output "$PAIR_WORK_ROOT/receipts/drive-before.json"

"$PAIR_PYTHON" "$PAIR_BUNDLE/transfer.py" fetch --rclone-config "$RCLONE_CONFIG" \
  --before "$PAIR_WORK_ROOT/receipts/drive-before.json" \
  --local-root "$PAIR_WORK_ROOT/metadata" \
  --report "$PAIR_WORK_ROOT/receipts/metadata-download.json"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PAIR_PYTHON" "$PAIR_BUNDLE/manifests.py" check-p01 \
  --metadata-root "$PAIR_WORK_ROOT/metadata" \
  --report "$PAIR_WORK_ROOT/receipts/p01-equivalence.json"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PAIR_PYTHON" "$PAIR_BUNDLE/manifests.py" create \
  --metadata-root "$PAIR_WORK_ROOT/metadata" --manifest-root "$PAIR_WORK_ROOT/manifests" \
  --report "$PAIR_WORK_ROOT/receipts/manifest-creation.json"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PAIR_PYTHON" "$PAIR_BUNDLE/manifests.py" verify \
  --metadata-root "$PAIR_WORK_ROOT/metadata" --manifest-root "$PAIR_WORK_ROOT/manifests" \
  --report "$PAIR_WORK_ROOT/receipts/manifest-verification.json"

# The initial publication was authorized by the issue. Reproducing selection
# does not need another upload; existing names are always refused.
if [[ "${PAIR_PUBLISH:-0}" == 1 ]]; then
  "$PAIR_PYTHON" "$PAIR_BUNDLE/transfer.py" publish --rclone-config "$RCLONE_CONFIG" \
    --before "$PAIR_WORK_ROOT/receipts/drive-before.json" \
    --local-root "$PAIR_WORK_ROOT/manifests" \
    --audit "$PAIR_WORK_ROOT/receipts/manifest-verification.json" \
    --report "$PAIR_WORK_ROOT/receipts/drive-publication.json"
fi
