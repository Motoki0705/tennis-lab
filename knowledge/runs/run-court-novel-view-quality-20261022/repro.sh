#!/usr/bin/env bash
set -euo pipefail
: "${SCENES_ROOT:?Set the read-only B00-B03 scene parent directory}"
: "${PROBE_OUTPUT_DIR:?Set a NEW plan/render output directory}"
: "${PROBE_REPLAY_DIR:?Set a different NEW plan replay directory}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCENES_ROOT PROBE_OUTPUT_DIR
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-}" in
  plan)
    .venv/bin/python "$SCRIPT_DIR/plan_probe.py" --scenes-root "$SCENES_ROOT" --output "$PROBE_OUTPUT_DIR"
    .venv/bin/python "$SCRIPT_DIR/plan_probe.py" --scenes-root "$SCENES_ROOT" --output "$PROBE_REPLAY_DIR"
    ;;
  enqueue)
    : "${TRAINING_QUEUE_DIR:?Set the shared main-tree queue directory}"
    : "${PROBE_SESSION:?Set current session ID}"
    printf -v probe_command '%q ' env "SCENES_ROOT=$SCENES_ROOT" "PROBE_OUTPUT_DIR=$PROBE_OUTPUT_DIR" \
      "PYTHONPATH=$PYTHONPATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      bash "$SCRIPT_DIR/render_stage.sh"
    bash .agents/skills/training-queue/scripts/training_queue.sh add "$probe_command" \
      --name court-novel-view-reproduction --provider codex --session "$PROBE_SESSION" --issue 1350 --resource all
    # Worker startup/observation uses the shared queue CLI; do not launch GPU work directly.
    bash .agents/skills/training-queue/scripts/training_queue.sh status
    ;;
  analyze)
    test -f "$PROBE_OUTPUT_DIR/render-timings.json"
    /usr/bin/time -v -o "$PROBE_OUTPUT_DIR/analysis-time.txt" .venv/bin/python "$SCRIPT_DIR/analyze_probe.py" --scenes-root "$SCENES_ROOT" --plan-root "$PROBE_OUTPUT_DIR"
    .venv/bin/python "$SCRIPT_DIR/review_bundle.py" --scenes-root "$SCENES_ROOT" --plan-root "$PROBE_OUTPUT_DIR" --replay-root "$PROBE_REPLAY_DIR"
    ;;
  *) echo 'usage: repro.sh plan | enqueue | analyze (see README.md)' >&2; exit 2 ;;
esac
