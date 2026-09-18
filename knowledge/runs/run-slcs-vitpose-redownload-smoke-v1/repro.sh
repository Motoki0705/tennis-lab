#!/usr/bin/env bash
# Enqueue a new smoke from the checkout containing this recorded driver.
# queue_repro.sh is the original queue capture; it omits this then-untracked driver.
set -euo pipefail
output_dir="${1:?Usage: bash repro.sh NEW_ABSOLUTE_OUTPUT_DIRECTORY SESSION_ID}"
provider_session="${2:?A new provider session ID is required}"
[[ "$output_dir" = /* && ! -e "$output_dir" ]] || { echo 'Use a fresh absolute output directory' >&2; exit 1; }
repo_root="$(git rev-parse --show-toplevel)"
common_git="$(git rev-parse --path-format=absolute --git-common-dir)"
main_repo="$(dirname "$common_git")"
cd "$repo_root"
export TRAINING_QUEUE_DIR="$main_repo/.training_queue"
printf -v smoke_command '%q ' env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  .venv/bin/python knowledge/runs/run-slcs-vitpose-redownload-smoke-v1/driver.py \
  --video "$main_repo/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000/media/cam0.mp4" \
  --observations "$main_repo/outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-004/video_000/clip_000/cam0_people.npz" \
  --checkpoint "$main_repo/third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth" \
  --installation "$main_repo/outputs/tennis_scene/analyze/vitpose_redownload/s42-001/installation.json" \
  --output "$output_dir"
bash .agents/skills/training-queue/scripts/training_queue.sh add "$smoke_command" \
  --name slcs-vitpose-redownload-smoke-repro --provider codex --session "$provider_session" --resource all
echo 'Enqueued. Start the shared training-queue worker when authorized.'
