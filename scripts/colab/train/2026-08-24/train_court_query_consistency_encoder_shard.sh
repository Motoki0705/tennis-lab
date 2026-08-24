#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for the current Issue #790 scaling-grid shard.
# New callers should use the dated 2026-08-25 entry point directly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[issue790] deprecated wrapper; forwarding to the 2026-08-25 scaling-grid shard." >&2
exec bash "${SCRIPT_DIR}/../2026-08-25/train_court_query_scaling_grid_shard.sh" "$@"
