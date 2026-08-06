"""Boundary tests for required PLCS scene scalar metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene


def test_scene_loader_requires_explicit_num_persons(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"fps": 30.0}))
    (tmp_path / "scalars.json").write_text(json.dumps({"num_cameras": 0}))
    np.save(tmp_path / "position.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(tmp_path / "rotation.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(
        tmp_path / "canonical_pose_3d.npy",
        np.zeros((1, 17, 3), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="num_persons"):
        load_scene(tmp_path)
