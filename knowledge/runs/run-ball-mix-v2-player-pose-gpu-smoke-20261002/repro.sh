#!/usr/bin/env bash
# Replay from an isolated checkout, inside the shared training queue.
# The original queue command is preserved in run.json and queue-captured-repro.sh.
# CUDA operator is explicitly rebuilt; its original byte hash is not assumed.
set -euo pipefail
: "${TENNIS_RUN_ID:?launch this replay through the shared GPU queue}"
[[ "${TENNIS_GPU_RESOURCE:-}" == all ]] || exit 2
REPO="${TENNIS_REPO:?set an isolated checkout; do not use the live campaign worktree}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
git checkout bd48b063d73778710f849e56178e07806343ec1e
BUILD_DIR="$(mktemp -d)"
bash "$REPO/tests/benchmarks/build_dino_extension.sh" /home/kamimura/projects/tennis-lab "$BUILD_DIR/operator"
PYTHONPATH="$BUILD_DIR/operator/lib:$REPO" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$REPO/.venv/bin/python" "$SCRIPT_DIR/replay_saved_smoke.py" "$BUILD_DIR/result.json"
