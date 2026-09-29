#!/usr/bin/env bash
# One exclusive queue job: all detector variants, raw scores/timings, tables and review video.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: $0 <asset-root> <completed-comparison.json> <new-report-with-plan.json>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
asset_root="$(cd "$1" && pwd)"
comparison="$2"
report="$3"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
bash "$code_root/tests/benchmarks/build_dino_extension.sh" "$asset_root" "$report/dino_extension"
export PYTHONPATH="$code_root:$report/dino_extension/lib"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/player_detection_far_diagnosis.py" \
    --phase infer --repo "$asset_root" --comparison "$comparison" --report "$report"
