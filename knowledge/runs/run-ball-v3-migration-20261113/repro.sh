#!/usr/bin/env bash
set -euo pipefail
source_root="${1:?pass the main repository root containing the retained legacy assets}"
code_commit="${MIGRATION_CODE_COMMIT:?set the execution commit recorded in the knowledge node}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
report_root="$source_root/outputs/ball_detection/migrate/video-v3/20261113"
.venv/bin/python "$script_dir/migrate.py" \
  --source-root "$source_root" --selection "$script_dir/selection.json" \
  --only-clip video_000/clip_000 --output "$report_root/dry-run.json"
.venv/bin/python "$script_dir/migrate.py" \
  --source-root "$source_root" --selection "$script_dir/selection.json" \
  --only-clip video_000/clip_000 --apply --version meiji-active-ball-v3-dry-20261113 \
  --code-commit "$code_commit" --output "$report_root/small-publication.json"
.venv/bin/python "$script_dir/migrate.py" \
  --source-root "$source_root" --selection "$script_dir/selection.json" \
  --apply --version meiji-active-ball-v3-20261113 --code-commit "$code_commit" \
  --output "$report_root/full-publication.json"
