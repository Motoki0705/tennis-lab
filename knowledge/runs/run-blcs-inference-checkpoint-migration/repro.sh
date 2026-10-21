#!/usr/bin/env bash
# Run this saved bundle against separately prepared, pinned checkouts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${MIGRATION_REPO:?Set MIGRATION_REPO to the new-code checkout}"
: "${ORIGINAL_REPO:?Set ORIGINAL_REPO to the original-code checkout}"
: "${MIGRATION_SOURCE:?Set MIGRATION_SOURCE to the original checkpoint}"
: "${MIGRATION_OUTPUT_ROOT:?Set MIGRATION_OUTPUT_ROOT to an unused absolute directory}"
test "$(git -C "$MIGRATION_REPO" rev-parse HEAD)" = 08ea733650aeb374f10871043313bc031a3b36a2
test "$(git -C "$ORIGINAL_REPO" rev-parse HEAD)" = 9f5fed173b9dd56e4a1bb72622ff2c771b8d1f66
case "$MIGRATION_OUTPUT_ROOT" in /*) ;; *) exit 2 ;; esac
test ! -e "$MIGRATION_OUTPUT_ROOT"
cd "$MIGRATION_REPO"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$SCRIPT_DIR/migrate.py" \
  --source "$MIGRATION_SOURCE" \
  --output "$MIGRATION_OUTPUT_ROOT/inference-v1.ckpt" \
  --original-checkout "$ORIGINAL_REPO" \
  --evidence-dir "$MIGRATION_OUTPUT_ROOT/evidence"
