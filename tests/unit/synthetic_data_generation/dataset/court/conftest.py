from __future__ import annotations

import math

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)


def _translation(x: float) -> RigidTransform:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return RigidTransform.from_matrix(matrix)


@pytest.fixture
def multi_court_layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((-8.0, 8.0)):
        scene_from_court = _translation(x)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.02},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id="court-0",
    )


@pytest.fixture
def captured_cameras() -> tuple[SceneCamera, ...]:
    intrinsics = (100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0)
    cameras = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        cameras.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=64,
                height=48,
                intrinsics=intrinsics,
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(cameras)


@pytest.fixture
def identity_metric_adapter() -> MetricSceneAdapter:
    return MetricSceneAdapter.from_nht_scene_from_metric_scene(
        np.eye(4, dtype=np.float64)
    )
