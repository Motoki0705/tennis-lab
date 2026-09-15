#!/usr/bin/env bash
# Resume the fixed Meiji L4 recipe from a trusted full-state Lightning checkpoint.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT [HYDRA_OVERRIDE ...]" >&2
    exit 2
fi
checkpoint="$1"
shift

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${gpu_name}" != "NVIDIA L4" ]]; then
    echo "This recipe requires one NVIDIA L4; assigned GPU: ${gpu_name}" >&2
    exit 2
fi
if [[ ! -d /content/drive/MyDrive ]]; then
    echo "Mount Google Drive at /content/drive before running this recipe." >&2
    exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
    echo "Resume checkpoint does not exist: ${checkpoint}" >&2
    exit 2
fi
checkpoint_root="$(cd "$(dirname "${checkpoint}")/../../../.." && pwd)"
checkpoint_relative="${checkpoint#"${checkpoint_root}/"}"
if [[ "${checkpoint_relative}" == "${checkpoint}" ]]; then
    echo "Checkpoint must be beneath its derived checkpoint root: ${checkpoint}" >&2
    exit 2
fi

exec .venv/bin/python -m src.tasks.ball_detection.scripts.train \
    --config-name train_meiji_3cam \
    "$@" \
    paths.checkpoint_root="${checkpoint_root}" \
    data.batch_size=8 \
    data.num_workers=8 \
    training.trainer.accumulate_grad_batches=4 \
    training.trainer.max_epochs=20 \
    run.gpus=1 \
    run.resume="${checkpoint_relative}" \
    run.test_after_fit=false
