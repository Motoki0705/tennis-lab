#!/usr/bin/env bash
# Run this replay as one resource=all job in the shared training queue.
# The exact original queue-generated command remains in queue-repro-original.sh.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "usage: repro.sh <new-output-directory> (from the intended repo checkout)" >&2
    exit 2
fi
: "${TENNIS_RUN_ID:?Run this replay through the shared training queue}"
: "${TENNIS_LAB_DINO_EXTENSION:?Select a verified DINO CUDA binary explicitly}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
git diff --exit-code f738a1918f966c6a9bc679f3a8e0432843f9ee62 -- \
    src/submodules/models/dino/person_detector.py src/submodules/models/dino/architecture.py
.venv/bin/python "$SCRIPT_DIR/prepare.py" \
    --clip "${TENNIS_DINO_CLIP:-data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clips/clip_000}" \
    --output "$1"
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python "$SCRIPT_DIR/compare.py" "$1"
