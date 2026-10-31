#!/usr/bin/env bash
set -euo pipefail

cd /content/tennis-lab
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
: "${RCLONE_CONFIG:?Colab must provide its authenticated rclone configuration}"

uv sync --locked --python 3.11
git submodule update --init third_party/dinov3
exec .venv/bin/python knowledge/runs/run-court-scene-pair-colab-s42/campaign.py run
