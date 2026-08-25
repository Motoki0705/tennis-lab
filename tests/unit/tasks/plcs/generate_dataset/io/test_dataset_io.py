"""PLCS normalized scene serialization contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.tasks.plcs.generate_dataset.scene_generator import SceneData
from src.utils.schema.court_normalization import (
    CourtCoordinateContractError,
    denormalize_court_position,
    normalize_court_position,
)


def _scene() -> tuple[SceneData, np.ndarray]:
    translation_m = np.asarray([[-4.0, -11.0, 1.0], [4.0, 11.0, 1.2]], dtype=np.float32)
    canonical = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3) / 100.0
    scene = SceneData(
        meta={
            "scene_id": "scene_000000",
            "motion_source": "test",
            "motion_category": "test",
            "gender": "neutral",
            "fps": 30.0,
            "num_frames": 2,
            "initial_position": [0.0, 0.0],
            "initial_yaw": 0.0,
            "num_cameras_sampled": 0,
        },
        position=normalize_court_position(translation_m),
        rotation=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (2, 1)),
        canonical_pose_3d=canonical,
        cameras=[],
        num_persons=1,
    )
    return scene, translation_m


def test_writer_persists_contract_and_preserves_canonical_metres(
    tmp_path: Path,
) -> None:
    scene, translation_m = _scene()
    path = PLCSDatasetWriter(tmp_path).save_scene(scene)
    loaded = load_scene(path)

    assert loaded["meta"]["court_coordinate_normalization"]["scale_xyz_m"] == [
        11.885,
        11.885,
        11.885,
    ]
    np.testing.assert_allclose(
        denormalize_court_position(loaded["position"]),
        translation_m,
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_array_equal(loaded["canonical_pose_3d"], scene.canonical_pose_3d)


def test_direct_loader_rejects_missing_scene_contract(tmp_path: Path) -> None:
    scene, _ = _scene()
    path = PLCSDatasetWriter(tmp_path).save_scene(scene)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("court_coordinate_normalization")
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(CourtCoordinateContractError, match="missing"):
        load_scene(path)
