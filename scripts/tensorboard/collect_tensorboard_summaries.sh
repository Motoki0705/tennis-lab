#!/usr/bin/env bash
# Collect all TensorBoard events.out.tfevents.* under runs/ and generate CSV +
# Markdown summaries. Skips processing if CSV/summary already exist for a
# given event file.

set -euo pipefail

# Directory where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default runs directory (can be overridden by first argument)
RUNS_DIR="${1:-runs}"

cd "${PROJECT_ROOT}"
exec uv run python scripts/tensorboard/collect_and_summarize.py --runs-dir "${RUNS_DIR}"
