#!/usr/bin/env bash
# Run this saved bundle against separately prepared, pinned checkouts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
: "${MIGRATION_REPO:?Set MIGRATION_REPO to the new-code checkout}"
: "${ORIGINAL_REPO:?Set ORIGINAL_REPO to the original-code checkout}"
: "${MIGRATION_SOURCE:?Set MIGRATION_SOURCE to the original checkpoint}"
: "${MIGRATION_OUTPUT_ROOT:?Set MIGRATION_OUTPUT_ROOT to an unused absolute directory}"
test "$(git -C "$MIGRATION_REPO" rev-parse HEAD)" = 44a6c978d7726e63f4a58cfbc0a99bd4cdf6a2bd
test "$(git -C "$ORIGINAL_REPO" rev-parse HEAD)" = e848e529928d2c8ee760faf0cd70112e2f976f13
case "$MIGRATION_OUTPUT_ROOT" in /*) ;; *) exit 2 ;; esac
test ! -e "$MIGRATION_OUTPUT_ROOT"
cd "$MIGRATION_REPO"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$SCRIPT_DIR/migrate.py" \
  --source "$MIGRATION_SOURCE" \
  --output "$MIGRATION_OUTPUT_ROOT/inference-v1.ckpt" \
  --original-checkout "$ORIGINAL_REPO" \
  --evidence-dir "$MIGRATION_OUTPUT_ROOT/evidence"
