#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
Q=.agents/skills/training-queue/scripts/training_queue.sh
test "$(git rev-parse HEAD)" = caca7e69b71da55b8a4430b52d587e4c81992380
bash "$Q" status
python3 - <<'CHECK'
from pathlib import Path
q=Path('.training_queue')
assert not list((q/'running').glob('*.job')) and not list((q/'jobs').glob('*.job'))
CHECK
receipt=$(bash "$Q" add "timeout --foreground --signal=TERM --kill-after=30s 600 .venv/bin/python outputs/tennis_scene/visualize/player_view/blender-acceptance/resume_batched.py perf" --name i1727-c3-batched-perf --provider codex --session 01a1bbf4-360d-78e1-b9b8-acd8b98b0237 --issue 1727 --resource all --idempotency-key player-view-c3-20261109/perf-caca7e69b7/attempt-1)
printf '%s\n' "$receipt"
job=${receipt#queued: }
bash "$Q" serve --idle-timeout 1
bash "$Q" status
bash "$Q" list
tail -n 50 "$TRAINING_QUEUE_DIR/logs/${job%.job}.log"
[[ -f "$TRAINING_QUEUE_DIR/done/$job" ]]
[[ ! -f "$TRAINING_QUEUE_DIR/failed/$job" ]]
