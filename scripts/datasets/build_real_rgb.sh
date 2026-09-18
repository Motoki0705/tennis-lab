#!/usr/bin/env bash
# One command -> shared GPU queue -> observations, labels, RGB tokens and splits.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_root"
common_git="$(git rev-parse --path-format=absolute --git-common-dir)"
main_root="$(dirname -- "$common_git")"
export TRAINING_QUEUE_DIR="$main_root/.training_queue"
queue_script="$task_root/.agents/skills/training-queue/scripts/training_queue.sh"

execute=false
if [[ "${1:-}" == "--execute" ]]; then
  execute=true
  shift
fi
mode="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in meiji|broadcast|all) ;; *) echo "Usage: $0 {meiji|broadcast|all} [Hydra overrides for one source]" >&2; exit 2 ;; esac
if [[ "$mode" == all && $# -gt 0 ]]; then
  echo "For source-specific overrides, select meiji or broadcast." >&2
  exit 2
fi

if "$execute"; then
  if [[ -z "${TENNIS_RUN_ID:-}" || -z "${TENNIS_REPRO_DIR:-}" ]]; then
    echo "--execute requires the shared training queue reservation." >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="${TENNIS_RGB_GPU:-0}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  asset_override="paths.external_asset_root=$main_root/third_party"
  if [[ "$mode" == broadcast || "$mode" == all ]]; then
    .venv/bin/python -m scripts.analysis.import_broadcast_ball
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset --config-name build_broadcast_slcs_dataset "$asset_override" "$@"
  fi
  if [[ "$mode" == meiji || "$mode" == all ]]; then
    .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset "$asset_override" "$@"
  fi
  if [[ "$mode" == all ]]; then
    .venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
    .venv/bin/python -m src.tennis_scene.scripts.assemble_slcs_dataset
  fi
  exit 0
fi

printf -v command '%q ' env "TENNIS_RGB_GPU=${TENNIS_RGB_GPU:-0}" bash "$task_root/scripts/datasets/build_real_rgb.sh" --execute "$mode" "$@"
bash "$queue_script" add "$command" --name "slcs-real-rgb-build-$mode" \
  --provider "${TENNIS_QUEUE_PROVIDER:-human}" --session "${TENNIS_QUEUE_SESSION:-manual}" --resource all
worker_pid=""
if [[ -f "$TRAINING_QUEUE_DIR/worker.pid" ]]; then read -r worker_pid < "$TRAINING_QUEUE_DIR/worker.pid"; fi
if [[ ! "$worker_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$worker_pid" 2>/dev/null; then
  bash "$queue_script" start
fi
