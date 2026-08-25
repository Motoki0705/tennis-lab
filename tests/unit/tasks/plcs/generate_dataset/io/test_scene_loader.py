"""Boundary tests for required PLCS scene scalar metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _meta() -> dict[str, object]:
    return {
        "fps": 30.0,
        "court_coordinate_normalization": (court_coordinate_normalization_metadata()),
    }


def test_scene_loader_requires_explicit_num_persons(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps(_meta()))
    (tmp_path / "scalars.json").write_text(json.dumps({"num_cameras": 0}))
    np.save(tmp_path / "position.npy", np.zeros((1, 3), dtype=np.float32))
    np.save(tmp_path / "rotation.npy", np.zeros((1, 2), dtype=np.float32))
    np.save(
        tmp_path / "canonical_pose_3d.npy",
        np.zeros((1, 17, 3), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="num_persons"):
        load_scene(tmp_path)


def test_scene_loader_rejects_legacy_visible_filenames(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps(_meta()))
    (tmp_path / "scalars.json").write_text(
        json.dumps({"num_cameras": 1, "num_persons": 1, "cam_0_params": {}})
    )
    for name, shape in {
        "position": (1, 3),
        "rotation": (1, 2),
        "canonical_pose_3d": (1, 17, 3),
        "cam_0_human_kp_uv": (1, 17, 2),
        "cam_0_human_kp_visible": (1, 17),
        "cam_0_human_visibility_ratio": (),
        "cam_0_court_kp_uv": (1, 20, 2),
        "cam_0_court_kp_visible": (1, 20),
        "cam_0_court_visibility_count": (),
    }.items():
        np.save(tmp_path / f"{name}.npy", np.zeros(shape, dtype=np.float32))

    with pytest.raises(FileNotFoundError, match="human_kp_vis"):
        load_scene(tmp_path)
