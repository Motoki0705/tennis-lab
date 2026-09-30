#!/usr/bin/env bash
# One all-capacity queue job; run-17 outer timeout -k 10s 10790s includes everything.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <main-repo> <report-with-plan>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$code_root:$2/dino_extension/lib"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/association_feature_guard.py" --report "$2" -- \
    bash -euo pipefail -c '
        bash "$1/tests/benchmarks/build_dino_extension.sh" "$2" "$3/dino_extension"
        exec "$1/.venv/bin/python" "$1/tests/benchmarks/person_unseen.py" --phase execute --report "$3"
    ' unseen "$code_root" "$1" "$2"
