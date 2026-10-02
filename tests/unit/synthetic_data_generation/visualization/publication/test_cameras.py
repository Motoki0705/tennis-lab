"""Unit tests for strict publication camera collections."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization.publication import cameras
from src.synthetic_data_generation.visualization.publication.cameras import (
    load_captured_cameras,
)


def _camera(camera_id: str, x: float) -> SceneCamera:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = x
    return SceneCamera(
        camera_id=camera_id,
        source_frame_index=int(x),
        width=64,
        height=64,
        intrinsics=(50.0, 0.0, 32.0, 0.0, 50.0, 32.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(transform),
        image_path=f"images/{camera_id}.png",
    )


def test_captured_camera_loader_requires_complete_export_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _camera("cam-0", 0.0), _camera("cam-1", 1.0)
    export = SimpleNamespace(
        scene_id="scene-0",
        camera_ids=("cam-0", "cam-1"),
        cameras=(first, second),
    )
    monkeypatch.setattr(cameras, "validate_standard_scene_export", lambda _: export)
    adapter = MetricSceneAdapter.from_nht_scene_from_metric_scene(
        np.eye(4, dtype=np.float64)
    )

    with pytest.raises(ValueError, match="complete reconstruction camera order"):
        load_captured_cameras(
            Path("missing-scene.json"),
            scene_id="scene-0",
            camera_ids=("cam-1", "cam-0"),
            metric_adapter=adapter,
        )

    collection = load_captured_cameras(
        Path("missing-scene.json"),
        scene_id="scene-0",
        camera_ids=("cam-0", "cam-1"),
        metric_adapter=adapter,
    )
    assert collection.camera_ids == ("cam-0", "cam-1")
