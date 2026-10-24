set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
OUT=outputs/synthetic_data_generation/profile/court-assets-verify/renders-native
NHT_PY="$(uv tool dir)/nht/bin/python"
"$NHT_PY" tmp/court_assets_probe.py --scenes tmp/court-scene-assets-v1/scenes --output "$OUT" --working tmp/court-assets-colab-native-working --repeats 2
.venv/bin/python tmp/court_assets_visibility.py --scenes tmp/court-scene-assets-v1/scenes --renders "$OUT" --output tmp/court-assets-colab-native-visibility.json
mv tmp/court-assets-colab-native-visibility.json "$OUT/visibility.json"
mkdir -p "$OUT/source"
cp tmp/court_assets_probe.py tmp/court_assets_visibility.py tmp/court_assets_render_native_colab.sh "$OUT/source/"
