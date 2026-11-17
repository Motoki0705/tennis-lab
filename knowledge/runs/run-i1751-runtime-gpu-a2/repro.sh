#!/usr/bin/env bash
# Replay adapter; queue-repro.sh preserves the byte-exact historical queue script.
# Prepare both pinned worktrees first. This adapter does not change their HEADs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${TENNIS_RUN_ID:-}" ]]; then
  echo 'Run only through the shared training queue; GPU acceptance is paused pending parent judgement.' >&2
  exit 1
fi
REPO="${TENNIS_REPO:-/home/kamimura/projects/tennis-lab/.claude/worktrees/scene-inference-runtime}"
cd "$REPO"
mkdir -p outputs/tennis_scene/evaluate
REPLAY_DIR="$(mktemp -d "$REPO/outputs/tennis_scene/evaluate/i1751-a2-replay.XXXXXX")"
# Keep measurement output separate from the historical evidence bundle.
cp "$SCRIPT_DIR"/*.py "$REPLAY_DIR/"
cp "$SCRIPT_DIR/inputs.json" "$REPLAY_DIR/inputs.json"
# run_pair.py verifies both pinned source HEADs before starting either model.
# --- original training command ---
PYTHONHASHSEED=42 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.worktree-outputs/video-batch-baseline/tennis_scene/evaluate/video-batch-baseline-035037-20261116/toolchain/MultiScaleDeformableAttention.so .venv/bin/python -u "$REPLAY_DIR/run_pair.py"
