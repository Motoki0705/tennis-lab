#!/usr/bin/env bash
# Queue entry: compare COCO / configured player DINO on chat-player-v1 val and Meiji labels.
# Usage from the active worktree: player_detection_comparison.sh <main-repo-assets> <new-report-dir>
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 <main-repo-assets> <new-report-dir>" >&2
    exit 2
fi
code_root="$(git rev-parse --show-toplevel)"
asset_root="$(cd "$1" && pwd)"
mkdir -p "$2"
report="$(cd "$2" && pwd)"
python="${code_root}/.venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
bash "$code_root/tests/benchmarks/build_dino_extension.sh" "$asset_root" "$report/dino_extension"
export PYTHONPATH="$code_root:$report/dino_extension/lib"

# The deployed checkpoint and preprocessing have a single source: pipeline.yaml.
mapfile -t detector < <("$python" - "$code_root" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

cfg = OmegaConf.load(Path(sys.argv[1]) / "src/tennis_scene/configs/pipeline.yaml")
if cfg.people_models.detector != "dino":
    raise ValueError("Comparison requires the pipeline's DINO detector")
print(cfg.people_models.dino_checkpoint)
runtime = cfg.people_models.runtime.dino_detector
print(runtime.short_side)
print(runtime.max_long_side)
print(runtime.confidence)
PY
)
if [[ ${#detector[@]} -ne 4 ]]; then
    echo "Could not read the deployed detector configuration" >&2
    exit 1
fi
"$python" -m src.tasks.player_detection.scripts.evaluate \
    "paths.project_root=$code_root" "paths.data_root=$asset_root/data" \
    "paths.checkpoint_root=$asset_root/ckpt" "paths.external_asset_root=$asset_root/third_party" \
    "paths.output_root=$report" "paths.artifact_root=$report" "paths.cache_root=$report/cache" \
    "run.output_dir=chat_val" "evaluate.split=val" "evaluate.frame_stride=1" "evaluate.num_workers=2" \
    "+evaluate.checkpoints.player_ft=${detector[0]}" \
    "evaluate.input_size.short_side=${detector[1]}" "evaluate.input_size.max_long_side=${detector[2]}" \
    "evaluation.score_threshold=${detector[3]}"

"$python" "$code_root/tests/benchmarks/player_detection_clips.py" \
    --repo "$asset_root" --dataset "$asset_root/data/tennis_multivew/processed/meiji_3cam/dataset" \
    --report "$report/meiji"
