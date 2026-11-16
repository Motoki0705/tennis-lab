#!/usr/bin/env bash
# Public redacted copy of the queue-generated replay: dino-batch-api-20261116
# Original bytes are retained privately; see publication-provenance.json.
# provider=codex issue=1750 resource=all logical_slot=all
set -uo pipefail
export TENNIS_GPU_RESOURCE=all
export TENNIS_GPU_SLOT=all
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:-.}"
cd "$REPO" || { echo "[repro] cannot cd to $REPO" >&2; exit 1; }
echo "[repro] target commit: f738a1918f966c6a9bc679f3a8e0432843f9ee62 (branch codex/dino-batch-api)"
git checkout f738a1918f966c6a9bc679f3a8e0432843f9ee62 2>/dev/null || echo "[repro] WARN: checkout f738a1918f966c6a9bc679f3a8e0432843f9ee62 failed; using current HEAD"
PATCH="$SCRIPT_DIR/uncommitted.patch"
if [ -s "$PATCH" ]; then git apply "$PATCH" 2>/dev/null || echo "[repro] WARN: patch did not apply cleanly"; fi
# --- original training command ---
timeout -k 10s 900s env PYTHONPATH=. TENNIS_LAB_DINO_EXTENSION="$TENNIS_LAB_DINO_EXTENSION" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python outputs/tennis_scene/evaluate/dino_batch_api_20261116/compare.py outputs/tennis_scene/evaluate/dino_batch_api_20261116
