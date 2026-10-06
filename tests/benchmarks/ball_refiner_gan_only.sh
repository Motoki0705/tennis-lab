#!/usr/bin/env bash
# Enqueue both refiners with position loss decayed to zero and GAN capped at one.
set -euo pipefail
refiner_tag=${1:?usage: ball_refiner_gan_only.sh RUN_TAG CODEX_SESSION_ID}
refiner_session=${2:?An attributable session id is required}
[[ "$refiner_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run tag' >&2; exit 2; }
refiner_common=$(git rev-parse --path-format=absolute --git-common-dir)
export TRAINING_QUEUE_DIR="$(dirname "$refiner_common")/.training_queue"
refiner_queue=.agents/skills/training-queue/scripts/training_queue.sh
[[ -f data/ball_refiner/single_object/manifest.json ]] || { echo 'Complete shared dataset required' >&2; exit 2; }
for refiner_dimensions in 2 3; do
  refiner_issue=991
  [[ "$refiner_dimensions" == 3 ]] && refiner_issue=1014
  refiner_name="i${refiner_issue}-rope-${refiner_dimensions}d-gan-only-eventonly-s42-${refiner_tag}"
  refiner_command="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates training=coordinate_gan_only model.dimensions=${refiner_dimensions} run.output_dir=ball_refiner/train/rope-${refiner_dimensions}d-gan-only-eventonly/${refiner_tag}-s42"
  bash "$refiner_queue" add "$refiner_command" --name "$refiner_name" --provider codex --session "$refiner_session" --issue "$refiner_issue" --resource all
done
