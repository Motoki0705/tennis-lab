#!/usr/bin/env bash
# One exclusive GPU cache job; launch only through the shared training queue.
set -euo pipefail
[[ $# -eq 5 ]] || { echo 'usage: <plan directory> <scene.yaml> <freeze.json> <extension lib> <plan sha256>' >&2; exit 2; }
report="$1"
scene="$2"
freeze="$3"
extension="$4"
plan_sha256="$5"
for path in "$report" "$scene" "$freeze" "$extension"; do
    [[ "$path" == /* ]] || { echo 'All paths must be absolute' >&2; exit 2; }
done
task_repo_root="$(git rev-parse --show-toplevel)"
python="$task_repo_root/.venv/bin/python"
export PYTHONPATH="$extension:$task_repo_root"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mapfile -t inputs < <("$python" - "$report/plan.json" <<'PY'
import json
import sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
print(plan['identity']['store']['directory'])
print(plan['identity']['evidence']['directory'])
PY
)
[[ ${#inputs[@]} -eq 2 ]] || { echo 'Invalid input plan' >&2; exit 2; }
timeout -k 10s 43190s "$python" "$task_repo_root/tests/benchmarks/association_feature_guard.py" --report "$report" -- \
    "$python" -m src.tasks.ball_refiner.scripts.meiji_context generate \
    --store "${inputs[0]}" --evidence "${inputs[1]}" --output "$report" \
    --scene-config "$scene" --freeze "$freeze" --plan-sha256 "$plan_sha256"
