#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON="$REPO/.venv/bin/python"

V3_ROOT="${MEIJI_V3_ROOT:-$REPO/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v3}"
V4_ROOT="${MEIJI_V4_ROOT:-$REPO/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4}"
V1_DIR="${MEIJI_V1_DIR:-$REPO/data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events}"
V2_DIR="${MEIJI_V2_DIR:-$REPO/data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events_v2}"
OUTPUT_JSON="${1:-$REPO/outputs/meiji-event-labels-v4/verification.json}"

cd "$REPO"
"$PYTHON" -m src.tennis_scene.scripts.build_meiji_event_benchmark \
  --version v4 \
  --base-root "$V3_ROOT" \
  --annotations-dir "$V2_DIR" \
  --output "$V4_ROOT"

"$PYTHON" "$SCRIPT_DIR/verify_multiview_v4.py" \
  --v3-root "$V3_ROOT" \
  --v4-root "$V4_ROOT" \
  --v1-dir "$V1_DIR" \
  --v2-dir "$V2_DIR" \
  --output "$OUTPUT_JSON"
