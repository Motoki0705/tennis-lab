#!/usr/bin/env bash
set -euo pipefail
task_repo_root="$(git rev-parse --show-toplevel)"
task_assets_root="${TENNIS_LAB_LOCAL_ASSET_ROOT:-$task_repo_root}"
task_python="${TENNIS_LAB_PYTHON:-$task_assets_root/.venv/bin/python}"
cd "$task_repo_root"
"$task_python" -m src.tennis_scene.scripts.align_gvhmr_world \
  --clip-dir "$task_assets_root/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000" \
  --motion-dir "$task_repo_root/knowledge/runs/run-plcs-gvhmr-similarity-clip000/inputs" \
  --asset-repository-root "$task_assets_root" \
  --body-models-dir "$task_assets_root/third_party/GVHMR/inputs/checkpoints/body_models" \
  --output-dir "${TENNIS_LAB_ALIGNMENT_OUTPUT_DIR:-$task_repo_root/outputs/plcs_gvhmr_similarity/reproduction}" \
  --render-video
