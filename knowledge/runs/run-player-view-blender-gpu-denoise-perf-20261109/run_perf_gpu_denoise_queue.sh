#!/usr/bin/env bash
set -euo pipefail
cd /content/tennis-lab
export TRAINING_QUEUE_DIR=/content/tennis-lab/.training_queue
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
Q=.agents/skills/training-queue/scripts/training_queue.sh
test "$(git rev-parse HEAD)" = 993c7e2310a76ebc7145ca33b9698be5b4996ddc
bash "$Q" status
python3 - <<'CHECK'
from pathlib import Path
q=Path('.training_queue')
assert not list((q/'running').glob('*.job')) and not list((q/'jobs').glob('*.job'))
CHECK
receipt=$(bash "$Q" add "timeout --foreground --signal=TERM --kill-after=30s 600 .venv/bin/python outputs/tennis_scene/visualize/player_view/blender-acceptance/gpu_denoise.py perf" --name i1727-c3-gpu-denoise-perf --provider codex --session 01a1bbf4-360d-78e1-b9b8-acd8b98b0237 --issue 1727 --resource all --idempotency-key player-view-c3-20261109/perf-gpu-denoise-993c7e2310/attempt-1)
printf '%s\n' "$receipt"
job=${receipt#queued: }
bash "$Q" serve --idle-timeout 1
bash "$Q" status
bash "$Q" list
tail -n 50 "$TRAINING_QUEUE_DIR/logs/${job%.job}.log"
[[ -f "$TRAINING_QUEUE_DIR/done/$job" ]]
[[ ! -f "$TRAINING_QUEUE_DIR/failed/$job" ]]
