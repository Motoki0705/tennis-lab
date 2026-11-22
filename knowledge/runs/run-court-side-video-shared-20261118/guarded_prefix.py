import sys
from pathlib import Path

from health_guard import run_variant

root = Path(__file__).resolve().parent
common = root.parents[3]
old = Path("/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/x3d-dpt-20261116/meiji-four-camera-final/x3d/clip_018/store")
command = [sys.executable, str(root / "collect_prefix.py"), "--run", str(root), "--outputs", str(common), "--old-x3d", str(old)]
result = run_variant(root, "prefix", command, Path.cwd())
print(result, flush=True)
raise SystemExit(0 if result["status"] == "complete" else 1)
