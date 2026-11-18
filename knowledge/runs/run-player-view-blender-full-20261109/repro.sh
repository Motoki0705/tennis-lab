#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${TENNIS_REPO:?Set TENNIS_REPO to a fresh tennis-lab checkout}
task_script_dir="$REPO/outputs/tennis_scene/visualize/player_view/blender-acceptance"
mkdir -p "$task_script_dir"
(
  cd "$task_script_dir"
  for name in run_full_gpu_denoise_queue.sh gpu_denoise.py; do
    if [[ -e "$name" ]]; then
      cmp "$SCRIPT_DIR/$name" "$name"
    else
      cp "$SCRIPT_DIR/$name" .
    fi
  done
)
(
  cd "$task_script_dir"
  mkdir -p operations
  for name in performance_gate_gpu_denoise.json prefix_manifest_gpu_denoise.json; do
    if [[ -e "operations/$name" ]]; then
      cmp "$SCRIPT_DIR/$name" "operations/$name"
    else
      cp "$SCRIPT_DIR/$name" operations/
    fi
  done
)
bash "$SCRIPT_DIR/colab-original-repro.sh"
python3 - "$REPO" <<'REPORTS'
from pathlib import Path
import hashlib,json,os,shutil,sys
root=Path(sys.argv[1])/'outputs/tennis_scene/visualize/player_view/blender-acceptance/full-gpu-denoise'
manifest=json.loads((root/'artifacts/manifest.json').read_text())
for identity in (0,1):
    source=root/'work/assembled'/f'player_{identity}/render.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['render_reports'][str(identity)]
    temporary=root/'work'/f'render_player_{identity}.json.delivery'
    shutil.copyfile(source,temporary)
    os.replace(temporary,root/'artifacts'/f'render_player_{identity}.json')
REPORTS
