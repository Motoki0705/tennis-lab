#!/usr/bin/env bash
set -euo pipefail
TRIANGULATION_PYTHON="${TRIANGULATION_PYTHON:-/home/kamimura/projects/tennis-lab/.venv/bin/python}"
TRIANGULATION_CLIP="${TRIANGULATION_CLIP:-/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000}"
TRIANGULATION_OUTPUT="${TRIANGULATION_OUTPUT:-outputs/triangulation/meiji-clip000}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
"$TRIANGULATION_PYTHON" -m scripts.analysis.triangulation.evaluate --clip "$TRIANGULATION_CLIP" --output "$TRIANGULATION_OUTPUT"
"$TRIANGULATION_PYTHON" -m scripts.analysis.triangulation.render --clip "$TRIANGULATION_CLIP" --output "$TRIANGULATION_OUTPUT"
