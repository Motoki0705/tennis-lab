#!/usr/bin/env bash
# The queue wraps build + extraction in timeout -k 10s 7190s (total <= 2 h).
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <main-repo> <report-with-plan.json>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash "$code_root/tests/benchmarks/build_dino_extension.sh" "$1" "$2/dino_extension"
export PYTHONPATH="$code_root:$2/dino_extension/lib"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/association_feature_guard.py" --report "$2" -- \
    "$code_root/.venv/bin/python" "$code_root/tests/benchmarks/association_recalibration_features.py" \
    --phase extract --repo "$1" --report "$2"
