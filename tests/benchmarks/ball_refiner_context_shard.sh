#!/usr/bin/env bash
# One pinned whole-clip job. Always invoke through the shared training queue.
set -euo pipefail
if [[ $# -ne 8 ]]; then
    echo "usage: $0 <asset_root> <detector_cache> <plan.json> <scene.yaml> <extension_dir> <shard_index> <new_context> <new_report>" >&2
    exit 2
fi
asset_root="$1"
evidence="$2"
plan="$3"
scene="$4"
extension="$5"
shard_index="$6"
context="$7"
report="$8"
for path in "$asset_root" "$evidence" "$plan" "$scene" "$extension" "$context" "$report"; do
    [[ "$path" == /* ]] || { echo "All paths must be absolute" >&2; exit 2; }
done
[[ "$shard_index" =~ ^[0-9]+$ ]] || { echo "Shard index must be nonnegative" >&2; exit 2; }
[[ ! -e "$context" && ! -e "$report" ]] || { echo "Output paths must be new" >&2; exit 2; }
code_root="$(git rev-parse --show-toplevel)"
python="$code_root/.venv/bin/python"
export PYTHONPATH="$extension/lib:$code_root"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
# Read the declared shard and cap, never choose them from outcomes or labels.
mapfile -t selection < <("$python" - "$plan" "$shard_index" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text())
print(plan['shards'][int(sys.argv[2])]['clip_id'])
print(plan['identity']['model_identity']['max_tracks'])
PY
)
[[ ${#selection[@]} -eq 2 ]] || { echo "Invalid plan selection" >&2; exit 2; }
mkdir -p "$(dirname "$report")"
mkdir "$report"
cp "$plan" "$report/plan.json"
cp "$scene" "$report/scene-context.yaml"
cp "$extension/build.json" "$report/build.json"
if [[ -n "${TENNIS_REPRO_DIR:-}" ]]; then
    cp "$report/plan.json" "$report/scene-context.yaml" "$report/build.json" "$TENNIS_REPRO_DIR/"
fi
"$python" -m src.tasks.ball_refiner.scripts.context_shards generate \
    --store "$asset_root/data/ball_detection/ball-mix-v1" --evidence "$evidence" \
    --output "$context" --scene-config "$scene" --max-tracks "${selection[1]}" \
    --plan "$plan" --shard-index "$shard_index"
CUDA_VISIBLE_DEVICES='' "$python" "$code_root/tests/benchmarks/ball_refiner_context.py" \
    --store "$asset_root/data/ball_detection/ball-mix-v1" --evidence "$evidence" \
    --context "$context" --report "$report/context-verification.json" \
    --pose-threshold 0.15 --clip-id "${selection[0]}"
if [[ -n "${TENNIS_REPRO_DIR:-}" ]]; then
    cp "$report/context-verification.json" "$TENNIS_REPRO_DIR/"
fi
