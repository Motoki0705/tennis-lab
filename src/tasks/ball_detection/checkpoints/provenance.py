"""Identity of the code that reads inputs, computes predictions and trains models."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT


def data_identity(manifest: Path) -> dict[str, Any]:
    """Record the snapshot and every clip's split, including clips with no play windows."""
    from src.tasks.ball_detection.data.store import BallFrameStore

    raw = json.loads(manifest.read_text())
    root = Path(raw["ball_store"]["directory"])
    for name, digest in raw["ball_store"]["hashes"].items():
        if dual_sha256(root / name) != digest:
            raise ValueError("Frozen snapshot identity changed")
    store = BallFrameStore(root)
    splits = {clip.clip_id: clip.split for clip in store.clips}
    return dict(manifest_sha256=dual_sha256(manifest), snapshot=raw["ball_store"], splits=splits,
                split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
                sampling=raw["sampling"], common_evaluation=raw["common_evaluation"])


def parent_provenance(saved: dict[str, Any]) -> dict[str, Any]:
    """Portable ancestry; loading descendants never opens ancestor files."""
    recipe = saved["recipe"]
    return dict(schema=saved["schema"], stage=saved["stage"], epoch=saved.get("epoch"),
                global_step=saved["global_step"], data=recipe.get("data_identity"),
                recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                code=saved.get("code"), parent=recipe.get("parent"))


def source_identity() -> dict[str, Any]:
    root = PROJECT_ROOT
    task = root / "src/tasks/ball_detection"
    directories = ("models/mdd", "preprocessing", "model_io", "data", "training", "evaluation", "checkpoints")
    paths = {path for directory in directories for path in (task / directory).rglob("*.py")}
    paths.update(root / name for name in (
        "src/tasks/ball_detection/configuration.py", "src/utils/configuration/cli.py",
        "src/tasks/ball_detection/scripts/train.py", "src/tasks/ball_detection/scripts/evaluate.py",
        "src/tasks/ball_detection/scripts/prepare_data.py",
        "src/utils/data/verification.py", "src/utils/data/file_ranges.py", "src/utils/checksum.py",
        "src/utils/models/dpt.py", "src/utils/models/blocks.py", "src/utils/models/components/ffn_layers.py",
        "src/utils/data/heatmaps.py", "src/tasks/base/training/losses.py"))
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return dict(directory=str(root), base_commit=commit, torch_version=str(torch.__version__),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(paths)})
