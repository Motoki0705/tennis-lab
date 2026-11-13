#!/usr/bin/env bash
# Re-run only input validation. This never launches GPT or acceptance.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git rev-parse --show-toplevel)"
common_git="$(git rev-parse --path-format=absolute --git-common-dir)"
main_repo="$(dirname -- "$common_git")"
asset_root="${CLIP000_ASSET_ROOT:-$main_repo/outputs/tennis_scene/evaluate/ball3d_comparison/clip000_20261103}"
video_dir="${CLIP000_VIDEO_DIR:-$repo_root/data/samples}"
report="${CLIP000_PREFLIGHT_REPORT:-$repo_root/outputs/ball_detection/evaluate/blank-strategy-comparison-20261113/preflight.json}"
cd -- "$repo_root"
PYTHONPATH=. .venv/bin/python "$SCRIPT_DIR/preflight.py" \
  --video-dir "$video_dir" --asset-root "$asset_root" --output "$report"
