"""Camera parsing must reuse the shared projector and reject bad parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.base.visualization.review.camera import (
    FRUSTUM_EDGES,
    parse_cameras,
)

DATA_ROOT = Path("/home/kamimura/projects/tennis-lab/data")
SCENE = DATA_ROOT / "blcs/single_object/scenes/scene_000000"

pytestmark = pytest.mark.skipif(
    not SCENE.is_dir(), reason="BLCS sample scene is unavailable"
)


def _scalars() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((SCENE / "scalars.json").read_text(encoding="utf-8"))
    return result


def _meta() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((SCENE / "meta.json").read_text(encoding="utf-8"))
    return result


def test_parse_cameras_reads_every_slot() -> None:
    cameras = parse_cameras(_scalars())
    assert len(cameras) == 6
    views = _meta()["court_keypoint_views"]
    for index, record in enumerate(cameras):
        assert record.id == f"cam_{index}"
        assert record.image_size == (1280, 720)
        assert record.center == pytest.approx(
            tuple(views[index]["camera_center_court_m"]), abs=1e-5
        )
        assert record.f == pytest.approx(1108.5125, abs=1e-2)


def test_intrinsics_and_camera_to_world_are_well_formed() -> None:
    record = parse_cameras(_scalars())[0]
    intrinsics = record.intrinsics()
    assert intrinsics.shape == (3, 3)
    assert intrinsics[2].tolist() == [0.0, 0.0, 1.0]
    transform = record.camera_to_world()
    assert transform.shape == (4, 4)
    assert transform[3].tolist() == [0.0, 0.0, 0.0, 1.0]
    rotation = transform[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_frustum_has_five_vertices_centred_on_the_camera() -> None:
    record = parse_cameras(_scalars())[2]
    vertices = record.frustum_vertices(depth=4.0)
    assert vertices.shape == (5, 3)
    assert np.allclose(vertices[0], np.asarray(record.center), atol=1e-6)
    # The four image-plane corners are 4 m in front of the camera along its axis.
    forward = np.asarray(vertices[1]) - np.asarray(record.center)
    assert np.linalg.norm(forward) > 4.0 - 1e-6
    assert set(np.asarray(FRUSTUM_EDGES).flatten().tolist()) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda scalars: scalars["cam_0_params"].pop("R"),
        lambda scalars: scalars["cam_0_params"].update({"w": 0}),
        lambda scalars: scalars["cam_0_params"].update({"C": [1.0, 2.0]}),
        lambda scalars: scalars["cam_0_params"].update(
            {"R": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}
        ),
        lambda scalars: scalars.update({"num_cameras": "6"}),
    ],
)
def test_malformed_cameras_are_rejected(mutate: Any) -> None:
    scalars = _scalars()
    mutate(scalars)
    with pytest.raises(ValueError):
        parse_cameras(scalars)
