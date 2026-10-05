#!/usr/bin/env bash
# Enqueue the predeclared 15-condition coordinate refiner comparison.
set -euo pipefail
refiner_tag=${1:?usage: ball_refiner_coordinates.sh RUN_TAG CODEX_SESSION_ID}
refiner_session=${2:?An attributable session id is required}
[[ "$refiner_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run tag' >&2; exit 2; }
refiner_common=$(git rev-parse --path-format=absolute --git-common-dir)
export TRAINING_QUEUE_DIR="$(dirname "$refiner_common")/.training_queue"
refiner_queue=.agents/skills/training-queue/scripts/training_queue.sh
[[ -f data/ball_refiner/single_object/manifest.json ]] || { echo 'Generate the complete shared dataset first' >&2; exit 2; }
for refiner_dimensions in 2 3; do
  for refiner_rate in 0.25 0.5 0.75; do
    for refiner_method in regression gan flow; do
      [[ "$refiner_dimensions" == 2 && "$refiner_method" == flow ]] && continue
      refiner_architecture=regression
      refiner_gan=0
      [[ "$refiner_method" == flow ]] && refiner_architecture=flow
      [[ "$refiner_method" == gan ]] && refiner_gan=0.002
      refiner_issue=991
      [[ "$refiner_dimensions" == 3 ]] && refiner_issue=1014
      refiner_name="i${refiner_issue}-coords-${refiner_dimensions}d-${refiner_method}-p${refiner_rate/./}-s42-${refiner_tag}"
      refiner_command="OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner.scripts.train_coordinates model.dimensions=${refiner_dimensions} model.architecture=${refiner_architecture} corruption.event_probability=${refiner_rate} training.gan_weight=${refiner_gan} run.output_dir=ball_refiner/train/coordinates-${refiner_dimensions}d-${refiner_method}-p${refiner_rate/./}/${refiner_tag}-s42"
      bash "$refiner_queue" add "$refiner_command" --name "$refiner_name" --provider codex --session "$refiner_session" --issue "$refiner_issue" --resource half
    done
  done
done
