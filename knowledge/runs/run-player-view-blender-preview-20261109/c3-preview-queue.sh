#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
SCRIPT=outputs/tennis_scene/visualize/player_view/blender-acceptance/c3-preview-queue.sh
OUT=outputs/tennis_scene/visualize/player_view/blender-acceptance/preview-renders
BLENDER=/content/blender/blender-5.2.1-linux-x64/blender
if [[ "${1:-}" == render ]]; then
    for player in 0 1; do
        "$BLENDER" --background --factory-startup --python-exit-code 1 --threads 4 \
            --python src/tennis_scene/visualization/blender/shots/player_view.py -- \
            --package outputs/tennis_scene/visualize/player_view/blender-acceptance/package \
            --output "$OUT/player_$player" --player-id "$player" --gpu-backend OPTIX \
            --source-frames 16 17 19 193 389 500 730 731 732 734 735 736
    done
    exit 0
fi
Q=.agents/skills/training-queue/scripts/training_queue.sh
bash "$Q" status
bash "$Q" list
python3 - <<'PYCHECK'
from pathlib import Path
import os
q=Path('.training_queue')
for state in ('jobs','running'):
    jobs=list((q/state).glob('*.job'))
    if jobs:
        raise SystemExit(f'Existing {state} queue jobs: {jobs}')
worker=q/'worker.pid'
if worker.exists():
    try:
        os.kill(int(worker.read_text().strip()), 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit('Existing queue worker; refusing duplicate serve')
PYCHECK
receipt=$(bash "$Q" add "timeout --signal=TERM --kill-after=30s 1500 bash $SCRIPT render" \
    --name i1727-c3-preview-24 --provider codex \
    --session 01a1bbf4-360d-78e1-b9b8-acd8b98b0237 --issue 1727 --resource all \
    --idempotency-key player-view-c3-20261109/c3-preview-24/attempt-1)
printf '%s\n' "$receipt"
job=${receipt#queued: }
[[ "$job" == *.job ]]
bash "$Q" serve --idle-timeout 1
bash "$Q" status
bash "$Q" list
tail -n 24 "$TRAINING_QUEUE_DIR/logs/${job%.job}.log"
[[ -f "$TRAINING_QUEUE_DIR/done/$job" ]]
[[ ! -f "$TRAINING_QUEUE_DIR/failed/$job" ]]
python3 - <<'PYVALIDATE'
from pathlib import Path
import json, struct
expected=[16,17,19,193,389,500,730,731,732,734,735,736]
root=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance/preview-renders')
for player in (0,1):
    path=root/f'player_{player}'
    data=json.loads((path/'render.json').read_text())
    assert data['complete'] is True
    assert data['player_id']==player
    assert data['engine']=='CYCLES'
    assert data['gpu_devices'] and all(d['type']=='OPTIX' and 'L4' in d['name'] for d in data['gpu_devices'])
    assert [f['source_frame'] for f in data['frames']]==expected
    files=sorted((path/'frames').glob('*.png'))
    assert len(files)==12
    for f in files:
        with f.open('rb') as handle:
            header=handle.read(24)
        assert header[:8]==b'\x89PNG\r\n\x1a\n'
    print(json.dumps({'player_id':player,'png_count':len(files),'blender_version':data['blender_version'],'gpu_devices':data['gpu_devices'],'render_json':str(path/'render.json')}))
print('C3_PREVIEW_VERIFIED_24_PNG')
PYVALIDATE
