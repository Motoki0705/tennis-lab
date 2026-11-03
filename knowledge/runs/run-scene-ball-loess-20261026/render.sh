#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo 'usage: render.sh SCENE_NPZ VIDEOS_DIR CHECKPOINT_ROOT NEW_OUTPUT_DIR BLENDER_BIN' >&2
  exit 2
fi
task_repo_root="$(git rev-parse --show-toplevel)"
task_bundle="$task_repo_root/knowledge/runs/run-scene-ball-loess-20261026"
task_scene="$1"
task_videos="$2"
task_checkpoints="$3"
task_output="$4"
task_blender="$5"
mkdir "$task_output"
mkdir -p "$task_output/tmp" "$task_output/cache" "$task_output/mpl-cache"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$task_repo_root"
export MPLCONFIGDIR="$task_output/mpl-cache" XDG_CACHE_HOME="$task_output/cache" TMPDIR="$task_output/tmp"
"$task_repo_root/.venv/bin/python" "$task_repo_root/knowledge/runs/run-multiview-readme-gif-20261021/render.py" \
  --scene "$task_scene" \
  --video cam0 "$task_videos/cam0.mp4" --video cam1 "$task_videos/cam1.mp4" --video cam2 "$task_videos/cam2.mp4" \
  --smpl-faces "$task_checkpoints/body_models/smplh/neutral/model.npz" \
  --smpl-joint-regressor "$task_checkpoints/body_models/gvhmr/smpl_neutral_J_regressor.pt" \
  --output-dir "$task_output/frames" --width 960 --frame-step 6 --start-frame 6 > "$task_output/render.log" 2>&1
ffmpeg -n -hide_banner -v error -threads 2 -filter_complex_threads 1 \
  -framerate 10000/1001 -i "$task_output/frames/%05d.png" \
  -filter_complex '[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle' \
  -loop 0 -final_delay 3 "$task_output/multiview_reconstruction.gif"
"$task_repo_root/.venv/bin/python" "$task_repo_root/knowledge/runs/run-multiview-readme-gif-20261021/verify.py" \
  --frames "$task_output/frames" --gif "$task_output/multiview_reconstruction.gif" --output "$task_output/review"
"$task_repo_root/.venv/bin/python" "$task_bundle/export_blender.py" --scene "$task_scene" \
  --checkpoints "$task_checkpoints" --output "$task_output/blender_bundle" > "$task_output/export_blender.log" 2>&1
"$task_blender" --background --factory-startup --python-exit-code 1 --threads 4 \
  --python src/tennis_scene/visualization/blender/shots/reveal.py -- \
  --bundle "$task_output/blender_bundle" --output "$task_output/blender" \
  --samples 4 --resolution-percentage 50 --frame-step 6 --render > "$task_output/blender.log" 2>&1
"$task_blender" --background "$task_output/blender/tennis_reveal.blend" --python-exit-code 1 --threads 4 \
  --python src/tennis_scene/visualization/blender/shots/validate_reveal.py -- \
  --bundle "$task_output/blender_bundle" --report "$task_output/blender_validation.json" > "$task_output/blender_validation.log" 2>&1
ffmpeg -n -v error -threads 2 -filter_complex_threads 1 -framerate 10000/1001 -pattern_type glob \
  -i "$task_output/blender/frames/*.png" -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart "$task_output/reveal.mp4"
ffmpeg -n -v error -threads 2 -filter_complex_threads 1 -i "$task_output/reveal.mp4" \
  -filter_complex 'fps=8,scale=400:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer:bayer_scale=4' \
  -loop 0 "$task_output/blender_reveal_meiji_cam1.gif"
ffmpeg -v error -xerror -i "$task_output/reveal.mp4" -f null -
ffmpeg -v error -xerror -i "$task_output/blender_reveal_meiji_cam1.gif" -f null -
