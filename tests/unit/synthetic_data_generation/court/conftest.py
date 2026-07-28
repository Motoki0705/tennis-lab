"""Fixtures for multi-court labels and orbit sampling."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.court.layout import (
    CourtInstance,
    MultiCourtLayout,
)
from src.synthetic_data_generation.scene_contract import (
    SceneCamera,
    SimilarityTransform,
)


def _similarity(translation_x: float) -> SimilarityTransform:
    return SimilarityTransform(
        scale=1.0,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(translation_x, 0.0, 0.0),
    )


@pytest.fixture
def two_court_layout() -> MultiCourtLayout:
    """Return two parallel courts separated laterally by 15 metres."""
    first_scene = _similarity(0.0)
    second_scene = _similarity(15.0)
    return MultiCourtLayout(
        geometry_artifact_fingerprint="a" * 64,
        reference_court_instance_id="court_0",
        courts=(
            CourtInstance(
                court_instance_id="court_0",
                candidate_id="court-0",
                scene_from_court=first_scene,
                court_from_scene=first_scene.inverse(),
                template_score=1.0,
            ),
            CourtInstance(
                court_instance_id="court_1",
                candidate_id="court-1",
                scene_from_court=second_scene,
                court_from_scene=second_scene.inverse(),
                template_score=0.8,
            ),
        ),
    )


def look_at(
    center: tuple[float, float, float],
    target: tuple[float, float, float],
) -> NDArray[np.float64]:
    """Return an OpenCV camera-to-world look-at matrix."""
    centre = np.asarray(center, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.column_stack((right, down, forward))
    result[:3, 3] = centre
    return result


def make_camera(
    camera_id: str,
    center: tuple[float, float, float],
    target: tuple[float, float, float] = (7.5, 0.0, 0.0),
) -> SceneCamera:
    """Build a wide-angle synthetic captured camera."""
    pose = look_at(center, target)
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="unit-fixture",
        image_uri=f"{camera_id}.png",
        source_frame_index=int(camera_id.rsplit("_", maxsplit=1)[-1]),
        group_id=0,
        width=640,
        height=480,
        intrinsics=(260.0, 0.0, 320.0, 0.0, 260.0, 240.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


@pytest.fixture
def captured_cameras() -> tuple[SceneCamera, ...]:
    """Return a sparse circular captured trajectory around two courts."""
    result = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)):
        center = (
            7.5 + 18.0 * float(np.cos(angle)),
            17.0 * float(np.sin(angle)),
            2.2,
        )
        result.append(make_camera(f"camera_{index}", center))
    return tuple(result)


@pytest.fixture
def support_points_scene() -> NDArray[np.float64]:
    """Return scene support far below the captured/orbit cameras."""
    x, y = np.meshgrid(
        np.linspace(-15.0, 30.0, 32),
        np.linspace(-25.0, 25.0, 36),
    )
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
