set -euo pipefail
OUT=outputs/synthetic_data_generation/profile/court-assets-verify/inputs
mkdir -p "$OUT"
TIMEFORMAT="elapsed_seconds=%R"
time .venv/bin/python .agents/skills/tennis-drive/scripts/drive.py download staging/court-scene-assets-a9281d9a41e39fc9 tmp/court-scene-assets-v1 --verify > /content/.court-assets-download.json
mv /content/.court-assets-download.json "$OUT/drive-download.json"
time .venv/bin/python -m src.synthetic_data_generation.scripts.court_scene_assets verify --root tmp/court-scene-assets-v1 --manifest-sha256 a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b > /content/.court-assets-manifest.json
mv /content/.court-assets-manifest.json "$OUT/manifest-verification.json"
cat "$OUT/manifest-verification.json"
