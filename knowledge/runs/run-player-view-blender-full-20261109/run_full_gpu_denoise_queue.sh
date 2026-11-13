#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
ROOT=outputs/tennis_scene/visualize/player_view/blender-acceptance
Q=.agents/skills/training-queue/scripts/training_queue.sh
test "$(git rev-parse HEAD)" = 993c7e2310a76ebc7145ca33b9698be5b4996ddc
test -z "$(git diff --name-only)"
test "$(sha256sum "$ROOT/gpu_denoise.py" | cut -d ' ' -f 1)" = e4940480a47ed3d200e71e56da113bbc0d600e16bbc0ad3541babf242a14c9ea
test ! -e "$ROOT/full-gpu-denoise"
python3 - <<'CHECK'
from pathlib import Path
import json
q=Path('.training_queue')
assert not list((q/'running').glob('*.job')) and not list((q/'jobs').glob('*.job'))
p=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance/operations/performance_gate_gpu_denoise.json')
assert json.loads(p.read_text())['passed'] is True
CHECK
remaining=$(python3 - <<'TIME'
from datetime import datetime,timezone
print(int((datetime.fromisoformat('2026-11-08T20:19:37.670427+00:00')-datetime.now(timezone.utc)).total_seconds()))
TIME
)
budget=$((remaining - 960))
(( budget > 60 ))
command="timeout --foreground --signal=TERM --kill-after=30s $budget .venv/bin/python $ROOT/gpu_denoise.py full"
python3 - "$remaining" "$budget" "$command" <<'BUDGET'
from pathlib import Path
from datetime import datetime,timezone
import sys,json
value={'at':datetime.now(timezone.utc).isoformat(),'gpu_deadline':'2026-11-08T20:19:37.670427+00:00','gpu_remaining_seconds':int(sys.argv[1]),'timeout_seconds':int(sys.argv[2]),'reserve_seconds':960,'command':sys.argv[3]}
p=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance/operations/gpu-denoise-command-budget.json')
p.write_text(json.dumps(value,indent=2)+chr(10))
print('C3_GPU_DENOISE_BUDGET '+json.dumps(value),flush=True)
BUDGET
receipt=$(bash "$Q" add "$command" --name i1727-c3-full-gpu-denoise --provider codex --session 01a1bbf4-360d-78e1-b9b8-acd8b98b0237 --issue 1727 --resource all --idempotency-key player-view-c3-20261109/gpu-denoise-993c7e2310/attempt-1)
printf '%s\n' "$receipt"
job=${receipt#queued: }
bash "$Q" serve --idle-timeout 1
bash "$Q" status
bash "$Q" list
tail -n 45 "$TRAINING_QUEUE_DIR/logs/${job%.job}.log"
[[ -f "$TRAINING_QUEUE_DIR/done/$job" ]]
[[ ! -f "$TRAINING_QUEUE_DIR/failed/$job" ]]
