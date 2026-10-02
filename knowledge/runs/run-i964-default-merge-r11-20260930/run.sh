#!/usr/bin/env bash
set -euo pipefail
code_root=$(git rev-parse --show-toplevel)
repo_root=/home/kamimura/projects/tennis-lab
export PYTHONPATH="$code_root"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$code_root/.venv/bin/python" "$code_root/tests/benchmarks/person_tracking_merge.py" \
  --repo "$repo_root" \
  --features "$repo_root/outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930" \
  --aflink "$repo_root/outputs/person_tracking/evaluate/tracker_matrix/i964-linking-r9-20260930/resources/AFLink_epoch20.pth" \
  --previous "$repo_root/outputs/person_tracking/evaluate/tracker_matrix/i964-hybrids-r10-20260930" \
  --report "${MERGE_REPORT_ROOT:-$repo_root/outputs/person_tracking/evaluate/tracker_matrix/i964-default-merge-r11-20260930}" \
  --phase "${1:?track, evaluate or audit}"
