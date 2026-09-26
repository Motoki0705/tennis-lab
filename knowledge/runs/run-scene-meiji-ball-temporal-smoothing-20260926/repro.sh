#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: repro.sh <committed-scene.json> <output-directory>" >&2
  exit 2
fi

source_scene_index=$(realpath "$1")
review_dir=$(realpath -m "$2")
checkout_root=$(git rev-parse --show-toplevel)
main_root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
python_bin="$main_root/.venv/bin/python"

cd "$checkout_root"
"$python_bin" -m scripts.compare_ball_smoothing --scene-index "$source_scene_index" --output-dir "$review_dir"

for method in savgol robust_spline ballistic_rts; do
  "$python_bin" -m src.tennis_scene.scripts.visualization \
    "paths.project_root=$main_root" \
    "paths.data_root=$main_root/data" \
    "paths.checkpoint_root=$main_root/ckpt" \
    "paths.external_asset_root=$main_root/third_party" \
    "paths.artifact_root=$review_dir" \
    "paths.output_root=$review_dir" \
    "input=$method/scene.npz" \
    "output=$method/mesh_full.mp4" \
    display=false start_frame=0 style.player_representation=smpl
  "$python_bin" -m src.tennis_scene.scripts.visualization \
    "paths.project_root=$main_root" \
    "paths.data_root=$main_root/data" \
    "paths.checkpoint_root=$main_root/ckpt" \
    "paths.external_asset_root=$main_root/third_party" \
    "paths.artifact_root=$review_dir" \
    "paths.output_root=$review_dir" \
    "input=$method/scene.npz" \
    output=null "preview_output=$method/mesh_preview.png" \
    display=false start_frame=500 style.player_representation=smpl
done
