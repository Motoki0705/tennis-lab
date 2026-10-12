"""Identity of the code that reads inputs, computes predictions and trains models."""
from __future__ import annotations

import hashlib
import subprocess
from typing import Any

import torch

from src.utils.paths import PROJECT_ROOT


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
