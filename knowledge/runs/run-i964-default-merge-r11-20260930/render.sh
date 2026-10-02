#!/usr/bin/env bash
set -euo pipefail
code_root=$(git rev-parse --show-toplevel)
report_root=${MERGE_REPORT_ROOT:-/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-default-merge-r11-20260930}
export PYTHONPATH="$code_root"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/person_tracking_linking_report.py" --report "$report_root" --run 11
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/person_tracking_merge_review.py" --report "$report_root"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/person_tracking_matrix_video.py" --report "$report_root" \
  --baseline strongsort_pp_pose_merge_off__clipreid_vitb16_market1501 \
  --candidate strongsort_pp_pose_merge_on__clipreid_vitb16_market1501 \
  --merge-audit "$report_root/merge_audit.json"
