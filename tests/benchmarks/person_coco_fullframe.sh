#!/usr/bin/env bash
# The queue command wraps this entire job in timeout 5400 (including build).
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: $0 <asset-root> <comparison.json> <report-with-plan.json>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
bash "$code_root/tests/benchmarks/build_dino_extension.sh" "$1" "$3/dino_extension"
export PYTHONPATH="$code_root:$3/dino_extension/lib"
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/person_coco_fullframe.py" \
    --phase infer --repo "$1" --comparison "$2" --report "$3"
