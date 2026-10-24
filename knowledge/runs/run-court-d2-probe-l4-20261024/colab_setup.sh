set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
export MAX_JOBS=2
COURT_PROBE_OUTPUT="$1"
COURT_PROBE_TIMER=knowledge/runs/run-court-scale-pilot-l4-20261023/timed_command.py
COURT_PROBE_ASSETS=data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
mkdir -p "$COURT_PROBE_OUTPUT/setup" tmp
git rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python3 "$COURT_PROBE_TIMER" --receipt "$COURT_PROBE_OUTPUT/setup/uv.json" -- uv sync --locked
python3 "$COURT_PROBE_TIMER" --receipt "$COURT_PROBE_OUTPUT/setup/nht.json" -- .venv/bin/python -m spin setup-nht --render-only
python3 "$COURT_PROBE_TIMER" --receipt "$COURT_PROBE_OUTPUT/setup/asset-download.json" -- .venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download "$COURT_PROBE_ASSETS" "$COURT_PROBE_ASSETS" --verify
python3 "$COURT_PROBE_TIMER" --receipt "$COURT_PROBE_OUTPUT/setup/asset-verify.json" -- .venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify --root "$COURT_PROBE_ASSETS" --manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
