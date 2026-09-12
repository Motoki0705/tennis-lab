#!/usr/bin/env bash
# Drive mounting is performed separately (or by the Colab workflow runner).
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "${gpu_name}" != "NVIDIA L4" ]]; then
    echo "This recipe requires one NVIDIA L4; assigned GPU: ${gpu_name}" >&2
    exit 2
fi
if [[ ! -d /content/drive/MyDrive ]]; then
    echo "Mount Google Drive at /content/drive before running this recipe." >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install uv
fi
uv sync --locked

# The catalog stages this file into data/. Notebook runs use the Drive source.
archive="${MEIJI_ARCHIVE:-/content/drive/MyDrive/tennis_lab/data/meiji_3cam.tar}"
.venv/bin/python scripts/colab/setup/prepare_meiji_archive.py \
    --archive "${archive}" \
    --destination data/tennis_multivew/processed/meiji_3cam/dataset

exec .venv/bin/python -m src.tasks.ball_detection.scripts.train_meiji_l4 \
    "paths.output_root=/content/drive/MyDrive/tennis_lab/outputs" \
    "run.output_dir=ball_detection/meiji_3cam_$(date -u +%Y%m%dT%H%M%SZ)" \
    "$@"
