#!/usr/bin/env bash
# Must be queued as resource=all with an outer timeout including build/validation.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <main-repo> <report-with-plan-and-preflight>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$code_root:$2/dino_extension/lib"
# Guard the build, execute, fresh-process reload and video as a single job.
"$code_root/.venv/bin/python" "$code_root/tests/benchmarks/association_feature_guard.py" --report "$2" -- \
    bash -euo pipefail -c '
        bash "$1/tests/benchmarks/build_dino_extension.sh" "$2" "$3/dino_extension"
        "$1/.venv/bin/python" "$1/tests/benchmarks/pipeline_qualification.py" --phase execute --report "$3"
        CUDA_VISIBLE_DEVICES="" "$1/.venv/bin/python" "$1/tests/benchmarks/pipeline_qualification.py" --phase validate --report "$3"
    ' qualification "$code_root" "$1" "$2"
