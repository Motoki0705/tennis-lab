#!/usr/bin/env bash
# Regenerate the recorded reveal with the current checkout. Historical commit,
# command and patch remain in run.json / uncommitted.patch; do not apply them
# over the current Blender module layout.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${TENNIS_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BUNDLE="${TENNIS_REVEAL_BUNDLE:-/home/kamimura/projects/tennis-lab/outputs/tennis_scene/visualize/blender_export/pr1203-meiji-cam1/bundle}"
OUTPUT="${TENNIS_REVEAL_OUTPUT:?Set TENNIS_REVEAL_OUTPUT to a new output directory}"
BLENDER_BIN="${BLENDER_BIN:-/home/kamimura/.cache/tennis-lab/blender/blender-5.2.1-linux-x64/blender}"
cd "$REPO"
if [[ -e "$OUTPUT" ]]; then
    echo "Output already exists: $OUTPUT" >&2
    exit 1
fi
BLENDER=(env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe "$BLENDER_BIN")
"${BLENDER[@]}" --background --factory-startup --python-exit-code 1 --threads 4 \
    --python src/tennis_scene/visualization/blender/shots/reveal.py -- \
    --bundle "$BUNDLE" --output "$OUTPUT" --samples 4 --resolution-percentage 50 --frame-step 4 --render
"${BLENDER[@]}" --background "$OUTPUT/tennis_reveal.blend" --python-exit-code 1 --threads 4 \
    --python src/tennis_scene/visualization/blender/shots/validate_reveal.py -- \
    --bundle "$BUNDLE" --report "$OUTPUT/validation.json"
ffmpeg -v error -framerate 15000/1001 -pattern_type glob -i "$OUTPUT/frames/*.png" \
    -c:v libx264 -crf 16 -pix_fmt yuv420p -movflags +faststart "$OUTPUT/reveal_preview.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration:format=duration,size \
    -of json "$OUTPUT/reveal_preview.mp4" > "$OUTPUT/video_metadata.json"
