set -euo pipefail
export PATH="$(uv tool dir --bin):$PATH"
export PYTHONPATH="$PWD"
COURT_PROBE_OUTPUT="$1"
COURT_PROBE_TIMER=knowledge/runs/run-court-scale-pilot-l4-20261023/timed_command.py
test ! -e "$COURT_PROBE_OUTPUT/probe"
.venv/bin/python "$COURT_PROBE_TIMER" --receipt "$COURT_PROBE_OUTPUT/render-timing.json" --sample-gpu -- .venv/bin/python -m cProfile -o "$COURT_PROBE_OUTPUT/profile.pstats" knowledge/runs/run-court-d2-support-cpu-20261024/render_probe.py --selection knowledge/runs/run-court-d2-support-cpu-20261024/render-selection.json --asset-root data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified --output "$COURT_PROBE_OUTPUT/probe"
.venv/bin/python - "$COURT_PROBE_OUTPUT" <<'PY'
import json
import pstats
import sys
from pathlib import Path

root = Path(sys.argv[1])
stats = pstats.Stats(str(root / "profile.pstats"))
names = {"evaluate_render", "encode_rgb_jpeg", "savez_compressed", "atomic_json", "reconstruct", "load_scene_inputs", "__call__", "_invoke", "validate", "wait", "_savez", "run", "write_bytes", "rename", "file_sha256"}
rows = [
    {"file": key[0], "line": key[1], "function": key[2], "primitive_calls": value[0], "calls": value[1], "self_seconds": value[2], "cumulative_seconds": value[3]}
    for key, value in stats.stats.items()
    if key[2] in names
]
temporary = root / "profile-summary.partial"
temporary.write_text(json.dumps({"total_seconds": stats.total_tt, "rows": rows}, indent=2) + "\n")
temporary.rename(root / "profile-summary.json")
PY
