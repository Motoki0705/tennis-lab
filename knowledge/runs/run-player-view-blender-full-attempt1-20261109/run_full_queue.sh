#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
ROOT=outputs/tennis_scene/visualize/player_view/blender-acceptance
Q=.agents/skills/training-queue/scripts/training_queue.sh
test "$(git rev-parse HEAD)" = 2791ae2fc3fe660c01b79d337ef46b0752aa0fa3
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"
test "$(sha256sum "$ROOT/render_full.py" | cut -d ' ' -f 1)" = 924d1252301567755c5b7b6ad5e01f4b7d56880ef58f71cfcd7c1170848957e4
test "$(sha256sum "$ROOT/package/manifest.json" | cut -d ' ' -f 1)" = 708affe5b630ba50014bf9f3f7a5ff4a730440906c1b1a4f99e920bfe5247280
test ! -e "$ROOT/full"
bash "$Q" status
bash "$Q" list
python3 - <<'PREFLIGHT'
from pathlib import Path
import os
q=Path('.training_queue')
for state in ('jobs','running'):
    jobs=list((q/state).glob('*.job'))
    if jobs:
        raise SystemExit(f'Existing {state} jobs; refusing a duplicate full run: {jobs}')
p=q/'worker.pid'
if p.exists():
    try:
        os.kill(int(p.read_text().strip()),0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit('Existing live queue worker; refusing duplicate serve')
PREFLIGHT
remaining=$(python3 - <<'BUDGET'
from datetime import datetime,timezone
print(int((datetime.fromisoformat('2026-11-08T20:19:37.670427+00:00')-datetime.now(timezone.utc)).total_seconds()))
BUDGET
)
budget=$((remaining - 960))
(( budget > 60 ))
command="timeout --foreground --signal=TERM --kill-after=30s $budget .venv/bin/python $ROOT/render_full.py --blender /content/blender/blender-5.2.1-linux-x64/blender --package $ROOT/package --output $ROOT/full"
python3 - "$remaining" "$budget" "$command" <<'RECEIPT'
from pathlib import Path
from datetime import datetime,timezone
import sys,json
record={'at':datetime.now(timezone.utc).isoformat(),'gpu_deadline':'2026-11-08T20:19:37.670427+00:00','gpu_remaining_seconds_at_registration':int(sys.argv[1]),'command_timeout_seconds':int(sys.argv[2]),'reserve_seconds':960,'command':sys.argv[3]}
p=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance/operations/full-command-budget.json')
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(record,indent=2)+'\n')
print('C3_FULL_BUDGET '+json.dumps(record),flush=True)
RECEIPT
receipt=$(bash "$Q" add "$command" --name i1727-c3-full-2020 --provider codex \
    --session 01a1bbf4-360d-78e1-b9b8-acd8b98b0237 --issue 1727 --resource all \
    --idempotency-key player-view-c3-20261109/full-2791ae2fc3/attempt-1)
printf '%s\n' "$receipt"
job=${receipt#queued: }
[[ "$job" == *.job ]]
bash "$Q" serve --idle-timeout 1
bash "$Q" status
bash "$Q" list
tail -n 45 "$TRAINING_QUEUE_DIR/logs/${job%.job}.log"
[[ -f "$TRAINING_QUEUE_DIR/done/$job" ]]
[[ ! -f "$TRAINING_QUEUE_DIR/failed/$job" ]]
python3 - <<'COMPLETE'
from pathlib import Path
p=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance/full/artifacts')
for name in ('player_0.mp4','player_1.mp4','comparison.mp4','comparison_short.mp4','checkpoint_manifest.json'):
    assert (p/name).is_file(), name
print('C3_FULL_QUEUE_COMPLETE',flush=True)
COMPLETE
