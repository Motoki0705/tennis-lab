"""Tests for explicit Matplotlib 3D camera-view conversion."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.utils.projection.camera_projector import make_look_at_camera
from src.utils.rendering.camera_view import (
    CameraView3DConfig,
    camera_from_scene_cameras,
    camera_to_matplotlib_angles,
    scene_cameras_from_scene,
)

pytestmark = pytest.mark.unit


def _camera_params(
    center: tuple[float, float, float] = (0.0, -20.0, 5.0),
) -> dict[str, object]:
    camera = make_look_at_camera(center, (0.0, 0.0, 0.5), hfov_deg=40.0)
    return {
        "C": camera.C.tolist(),
        "R": camera.R.tolist(),
        "f": camera.f,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.w,
        "h": camera.h,
    }


def test_config_parses_mapping_values() -> None:
    config = CameraView3DConfig.from_mapping(
        {
            "mode": "look_at",
            "center": [2, -30, 8],
            "look_at": [0, 1, 0.5],
            "roll_deg": 3,
            "projection": "orthographic",
            "hfov_deg": 42,
            "scene_camera_index": 2,
            "zoom": 1.5,
        }
    )

    assert config.mode == "look_at"
    assert config.center == (2.0, -30.0, 8.0)
    assert config.look_at == (0.0, 1.0, 0.5)
    assert config.roll_deg == 3.0
    assert config.projection == "orthographic"
    assert config.hfov_deg == 42.0
    assert config.scene_camera_index == 2
    assert config.zoom == 1.5


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mode": "automatic"}, "view_3d.mode"),
        ({"projection": "weak-perspective"}, "view_3d.projection"),
        ({"center": [0, 0]}, "view_3d.center"),
        ({"center": [0, 0, 0], "look_at": [0, 0, 0]}, "must be different"),
        ({"hfov_deg": 180}, "view_3d.hfov_deg"),
        ({"scene_camera_index": -1}, "scene_camera_index"),
        ({"zoom": 0}, "view_3d.zoom"),
    ],
)
def test_config_rejects_invalid_values(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CameraView3DConfig.from_mapping(override)


def test_look_at_conversion_uses_court_world_direction() -> None:
    camera = make_look_at_camera((0.0, -20.0, 5.0), (0.0, 0.0, 0.0))

    elev, azim, roll = camera_to_matplotlib_angles(camera)

    assert elev == pytest.approx(math.degrees(math.atan2(5.0, 20.0)))
    assert azim == pytest.approx(-90.0)
    assert roll == pytest.approx(0.0, abs=1e-5)


def test_scene_camera_adaptation_reads_saved_params() -> None:
    camera = camera_from_scene_cameras([{"params": _camera_params()}], 0)

    assert camera.C.tolist() == pytest.approx([0.0, -20.0, 5.0])
    assert camera.w == 1280
    assert camera.h == 720


def test_scene_camera_adaptation_preserves_extrinsic_roll() -> None:
    params = _camera_params()
    rotation = np.asarray(params["R"], dtype=np.float64)
    angle = math.radians(12.0)
    rolled = rotation.copy()
    rolled[0] = math.cos(angle) * rotation[0] + math.sin(angle) * rotation[1]
    rolled[1] = -math.sin(angle) * rotation[0] + math.cos(angle) * rotation[1]
    params["R"] = rolled.tolist()

    resolved = CameraView3DConfig(
        mode="scene_camera",
        roll_deg=3.0,
    ).resolve([{"params": params}])

    assert resolved.roll_deg == pytest.approx(15.0, abs=1e-4)
    assert resolved.focal_length == pytest.approx(
        cast(float, params["f"]) / (0.5 * cast(int, params["w"]))
    )


def test_scene_camera_mode_requires_metadata() -> None:
    config = CameraView3DConfig(mode="scene_camera")

    with pytest.raises(ValueError, match="requires scene camera metadata"):
        config.resolve()


def test_integrated_scene_reads_cameras_from_metadata() -> None:
    cameras = [{"params": _camera_params()}]
    scene = SimpleNamespace(metadata={"cameras": cameras})

    assert scene_cameras_from_scene(scene) is cameras


def test_scene_camera_mode_rejects_missing_parameters() -> None:
    config = CameraView3DConfig(mode="scene_camera")

    with pytest.raises(ValueError, match="missing required fields"):
        config.resolve([{"params": {"C": [0, 0, 1]}}])


def test_apply_sets_angles_and_projection_on_3d_axis() -> None:
    config = CameraView3DConfig.from_mapping(
        {
            "mode": "look_at",
            "center": [0.0, -20.0, 5.0],
            "look_at": [0.0, 0.0, 0.0],
            "roll_deg": 4.0,
            "projection": "perspective",
            "hfov_deg": 40.0,
        }
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        resolved = config.apply(ax)

        assert ax.elev == pytest.approx(resolved.elev_deg)
        assert ax.azim == pytest.approx(-90.0)
        assert ax.roll == pytest.approx(4.0)
        assert ax._focal_length == pytest.approx(1.0 / math.tan(math.radians(20.0)))
        assert np.linalg.norm(ax.get_box_aspect()) == pytest.approx(
            1.8294640721620434 * 25.0 / 24.0 * config.zoom
        )
        first_box_aspect = ax.get_box_aspect().copy()
        config.apply(ax)
        np.testing.assert_allclose(ax.get_box_aspect(), first_box_aspect)
    finally:
        plt.close(fig)


def test_apply_sets_orthographic_projection() -> None:
    config = CameraView3DConfig(projection="orthographic")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        resolved = config.apply(ax)

        assert resolved.focal_length is None
        assert math.isinf(ax._focal_length)
    finally:
        plt.close(fig)


def test_default_mode_explicitly_reproduces_matplotlib_camera() -> None:
    resolved = CameraView3DConfig().resolve()

    assert resolved.elev_deg == 30.0
    assert resolved.azim_deg == -60.0
    assert resolved.roll_deg == 0.0
    assert resolved.focal_length == pytest.approx(1.0)
    assert resolved.zoom == pytest.approx(1.3)
