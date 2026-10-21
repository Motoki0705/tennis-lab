#!/usr/bin/env bash
# Run from a checkout containing render.py at 7c504fe6891ab85fb965dd09efb913ca960db983.
# Usage: bash knowledge/runs/run-multiview-readme-gif-20261021/repro.sh \
#   /path/to/scene.npz /path/to/samples /path/to/ckpt /path/inside/checkout/new-output
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "usage: $0 SCENE_NPZ SAMPLES_DIR CHECKPOINT_ROOT NEW_OUTPUT_DIR" >&2
  exit 2
fi
task_repo_root="$(git rev-parse --show-toplevel)"
gif_scene="$1"
gif_videos="$2"
gif_checkpoints="$3"
gif_output="$4"
mkdir -p "$gif_output/tmp" "$gif_output/cache" "$gif_output/mpl-cache"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$task_repo_root"
export MPLCONFIGDIR="$gif_output/mpl-cache" XDG_CACHE_HOME="$gif_output/cache" TMPDIR="$gif_output/tmp"
"$task_repo_root/.venv/bin/python" "$task_repo_root/knowledge/runs/run-multiview-readme-gif-20261021/render.py" \
  --scene "$gif_scene" \
  --video cam0 "$gif_videos/cam0.mp4" --video cam1 "$gif_videos/cam1.mp4" --video cam2 "$gif_videos/cam2.mp4" \
  --smpl-faces "$gif_checkpoints/body_models/smplh/neutral/model.npz" \
  --smpl-joint-regressor "$gif_checkpoints/body_models/gvhmr/smpl_neutral_J_regressor.pt" \
  --output-dir "$gif_output/frames" --width 960 --frame-step 6 --start-frame 6 \
  > "$gif_output/render.log" 2>&1
# Source: 60000/1001 fps; frames 6..1008, one output per six source frames.
# Skip the initial display sample (frame 0), whose reconstruction masks are invalid.
# The final sample is frame 1008 of 1010: its final delay is 3 centiseconds.
ffmpeg -hide_banner -v error -threads 2 -filter_complex_threads 1 \
  -framerate 10000/1001 -i "$gif_output/frames/%05d.png" \
  -filter_complex '[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle' \
  -loop 0 -final_delay 3 "$gif_output/multiview_reconstruction.gif"
"$task_repo_root/.venv/bin/python" "$task_repo_root/knowledge/runs/run-multiview-readme-gif-20261021/verify.py" \
  --frames "$gif_output/frames" --gif "$gif_output/multiview_reconstruction.gif" --output "$gif_output/review"
