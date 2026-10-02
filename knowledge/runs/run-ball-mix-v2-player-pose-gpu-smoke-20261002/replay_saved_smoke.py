"""Replay the exact saved script, redirecting only its result file."""
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
from typing import Any
original_write = Path.write_text
target = Path(sys.argv[1])
source = Path(__file__).with_name("gpu_smoke.py")
def write_result(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
    destination = target if path.name == "gpu-smoke-20261002.json" else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return original_write(destination, data, *args, **kwargs)
with patch.object(Path, "write_text", write_result):
    runpy.run_path(str(source), run_name="__main__")
