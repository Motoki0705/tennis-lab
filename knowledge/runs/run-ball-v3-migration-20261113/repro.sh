#!/usr/bin/env bash
set -euo pipefail
source_root="${1:?pass the main repository root containing the retained legacy assets}"
storage_root="${2:?pass the physical repository data/output root on the data disk}"
code_commit="${MIGRATION_CODE_COMMIT:?set the execution commit recorded in the knowledge node}"
bundle="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
report_root="$storage_root/outputs/ball_detection/migrate/video-v3/20261113"
.venv/bin/python "$bundle/migrate.py" \
  --source-root "$source_root" --selection "$bundle/selection-reviewed.json" \
  --storage-root "$storage_root" --legacy-image-baseline "$bundle/relocated-legacy-baseline.json" \
  --only-clip video_000/clip_000 --output "$report_root/relocated-dry-run.json"
.venv/bin/python "$bundle/migrate.py" \
  --source-root "$source_root" --selection "$bundle/selection-reviewed.json" \
  --storage-root "$storage_root" --legacy-image-baseline "$bundle/relocated-legacy-baseline.json" \
  --only-clip video_000/clip_000 --apply --version meiji-active-ball-v3-resume-check-20261114 \
  --code-commit "$code_commit" --output "$report_root/small-relocated-publication.json"
.venv/bin/python "$bundle/migrate.py" \
  --source-root "$source_root" --selection "$bundle/selection-reviewed.json" \
  --storage-root "$storage_root" --legacy-image-baseline "$bundle/relocated-legacy-baseline.json" \
  --apply --version meiji-active-ball-v3-20261113 --code-commit "$code_commit" \
  --output "$report_root/full-publication.json"
.venv/bin/python "$bundle/summarize.py" --source-root "$source_root" --reports "$report_root"
