from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)

# Metric scene -> NHT scene: half scale, a yaw of 90 degrees, and an offset.
NHT_UNITS_PER_METRE = 0.5


def _similarity() -> NDArray[np.float64]:
    matrix: NDArray[np.float64] = np.eye(4)
    matrix[:3, :3] = NHT_UNITS_PER_METRE * np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    matrix[:3, 3] = (1.0, -2.0, 0.5)
    return matrix


@pytest.fixture
def adapter() -> MetricSceneAdapter:
    forward = _similarity()
    return MetricSceneAdapter(
        nht_scene_from_metric_scene=tuple(float(value) for value in forward.ravel()),
        metric_scene_from_nht_scene=tuple(
            float(value) for value in np.linalg.inv(forward).ravel()
        ),
        nht_scene_units_per_metre=NHT_UNITS_PER_METRE,
    )


@pytest.fixture
def layout() -> MultiCourtLayout:
    courts = []
    for court_id, x in (("court-000", 0.0), ("court-001", -15.0)):
        matrix = np.eye(4)
        matrix[:3, 3] = (x, 3.0, 0.0)
        scene_from_court = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=court_id,
                candidate_id=f"candidate-{court_id}",
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
        complex_bounds_scene=(-50.0, -50.0, -5.0, 50.0, 50.0, 50.0),
        primary_court_instance_id="court-000",
    )
