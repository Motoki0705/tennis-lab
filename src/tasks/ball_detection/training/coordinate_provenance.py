"""Record the actual implementation, including changes not yet in a commit."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any

import torch

from src.utils.paths import PROJECT_ROOT


def coordinate_source_identity() -> dict[str, Any]:
    root = PROJECT_ROOT
    task = root / "src/tasks/ball_detection"
    patterns = ("models/mdd_pose/*.py", "preprocessing/*.py", "model_io/mdd*.py", "data/coordinate*.py", "data/temporal_sampling.py",
                "data/pose_windows.py", "data/play*.py", "data/annotation_states.py", "data/store.py",
                "training/coordinate*.py", "scripts/train_mdd_pose.py", "scripts/prepare_mdd_training.py",
                "scripts/evaluate_mdd_coordinates.py")
    paths = sorted({path for pattern in patterns for path in task.glob(pattern)})
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return dict(directory=str(root), base_commit=commit, torch_version=str(torch.__version__),
                source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
