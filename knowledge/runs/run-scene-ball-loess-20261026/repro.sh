#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo 'usage: repro.sh SOURCE_STORE VIDEOS_DIR CHECKPOINT_ROOT NEW_OUTPUT_DIR BLENDER_BIN' >&2
  exit 2
fi
task_repo_root="$(git rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_store="$1"
task_videos="$2"
task_checkpoints="$3"
task_output="$4"
task_blender="$5"
mkdir "$task_output"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONPATH="$task_repo_root"
for task_replay in replay replay_repeat; do
  "$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/replay.py" \
    --store "$task_store" --videos "$task_videos" --output "$task_output/$task_replay" > "$task_output/$task_replay.log" 2>&1
done
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/compare.py" --store "$task_store" \
  --replay "$task_output/replay" --repeat "$task_output/replay_repeat" --output "$task_output/comparison"
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/review_extremes.py" \
  --replay "$task_output/replay" --output "$task_output/extremes"
bash "$SCRIPT_DIR/render.sh" "$task_output/replay/scene.npz" "$task_videos" "$task_checkpoints" "$task_output/render" "$task_blender"
# Replace the standalone public 3D demo with the exact 3D panel of this render.
ffmpeg -n -v error -threads 2 -filter_complex_threads 1 -framerate 10000/1001 \
  -i "$task_output/render/frames/%05d.png" \
  -filter_complex 'crop=640:540:320:48,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer:bayer_scale=4' \
  -loop 0 -final_delay 3 "$task_output/render/rich_scene_rendering.gif"
"$task_repo_root/.venv/bin/python" "$SCRIPT_DIR/verify_media.py" \
  --replay "$task_output/replay" --render "$task_output/render"
