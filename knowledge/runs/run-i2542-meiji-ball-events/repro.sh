#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(git -C "$bundle" rev-parse --show-toplevel)
main_repo=$(dirname -- "$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)")
worktrees="$main_repo/.claude/worktrees"
python="$repo/.venv/bin/python"
refiner=${MEIJI_REFINER_CHECKOUT:-"$worktrees/parent-1722-issue-2567-primary"}
benchmark=${MEIJI_BENCHMARK:-"$main_repo/data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2"}
shared_outputs="$main_repo/.worktree-outputs"
checkpoint=${MEIJI_REFINER_CHECKPOINT:-"$shared_outputs/parent-1722-issue-2567-primary/ball_refiner_3d/train/regression/net-contacts-s42-v1/logs/version_0/checkpoints/best.ckpt"}
selection=${MEIJI_SELECTION:-"$shared_outputs/parent-1722-issue-2542-primary/tennis_scene/analyze/event-eval-selection/20261125T175807Z-e5e170/selection.json"}
rules=${MEIJI_RULE_RUN:-"$shared_outputs/parent-1722-issue-2568-primary/tennis_scene/analyze/rule_ball_events/meiji-multiview-v2-20261126"}
labels=${MEIJI_LABEL_ROOT:-"$shared_outputs"}
output=${1:-"$repo/outputs/tennis_scene/evaluate/meiji-ball-events/multiview-v2-20261126"}

if [[ "$output" != /* || "$benchmark" != /* || "$checkpoint" != /* || "$selection" != /* ]]; then
    printf '%s\n' 'Use absolute input and output paths.' >&2
    exit 2
fi
(
    cd -- "$refiner"
    "$python" -B "$bundle/predict.py" --benchmark "$benchmark" --checkpoint "$checkpoint" \
        --selection "$selection" --output "$output"
)
"$python" -B "$bundle/evaluate.py" --benchmark "$benchmark" --selection "$selection" \
    --rule-root "$rules" --labels-parent "$labels" --output "$output"
