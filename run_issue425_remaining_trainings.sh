#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${repo_root}/.venv/bin/python"
start_from="${ISSUE425_START_FROM:-plcs_multiview_axial_aug_off}"
trajectory_aug_on_resume="${ISSUE425_RESUME_TRAJECTORY_AUG_ON:-}"
has_started=0

if [[ -n "$trajectory_aug_on_resume" ]]; then
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

run_experiment() {
    local name="$1"
    shift

    if [[ "$has_started" -eq 0 ]]; then
        if [[ "$name" != "$start_from" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipping ${name} until ${start_from}"
            return
        fi
        has_started=1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${name}"
    (
        cd "$repo_root"
        "$python_bin" -u "$@"
    )
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${name}"
}

run_experiment \
    "plcs_multiview_axial_aug_off" \
    -m src.tasks.plcs.scripts.train \
    model=multiview_axial \
    data=multiview \
    training.trainer.max_epochs=50 \
    data.batch_size=1 \
    data.augmentation.enabled=false \
    run.output_dir=outputs/issue425/plcs_multiview_axial_aug_off

run_experiment \
    "trajectory_completion_aug_on" \
    -m src.tasks.trajectory_completion.scripts.train \
    data.batch_size=4 \
    ${trajectory_aug_on_resume:+run.resume=${trajectory_aug_on_resume}} \
    run.output_dir=outputs/issue425/trajectory_completion_aug_on

run_experiment \
    "trajectory_completion_aug_off" \
    -m src.tasks.trajectory_completion.scripts.train \
    data.batch_size=4 \
    data.augmentation.enabled=false \
    run.output_dir=outputs/issue425/trajectory_completion_aug_off

run_experiment \
    "event_detection_uv_aug_on" \
    -m src.tasks.event_detection.scripts.train_uv \
    data.batch_size=8 \
    run.output_dir=outputs/issue425/event_detection_uv_aug_on

run_experiment \
    "event_detection_uv_aug_off" \
    -m src.tasks.event_detection.scripts.train_uv \
    data.batch_size=8 \
    data.augmentation.enabled=false \
    run.output_dir=outputs/issue425/event_detection_uv_aug_off