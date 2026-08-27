"""Focused fixtures for canonical BLCS domain tests."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.composition import (
    GaussianAsset,
    GaussianAssetRole,
    GaussianCoordinates,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallGaussianSettings,
    BLCSCompositionAssets,
    BLCSTrack,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.camera_profiles import CameraProfileConfig
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)


@pytest.fixture
def blcs_assets() -> BLCSCompositionAssets:
    """Return one small deterministic metric ball asset contract."""
    return BLCSCompositionAssets(
        ball=GaussianAsset(
            asset_id="ball-surface",
            asset_class="ball",
            role=GaussianAssetRole.MOVABLE,
            coordinates=GaussianCoordinates.asset_local_metres(),
            gaussian_count=64,
            feature_dim=3,
            floating_dtype="float32",
            appearance_model="rgb",
            appearance_space="linear_rgb",
        ),
        settings=BLCSBallGaussianSettings(
            radius_m=0.0335,
            radial_scale_m=0.0018,
            tangential_scale_m=0.0048,
            opacity=0.94,
            base_color_linear_rgb=(0.72, 0.92, 0.08),
            seam_color_linear_rgb=(0.92, 0.95, 0.80),
            seam_width_radians=0.08,
            visibility_threshold=0.0001,
        ),
    )


@pytest.fixture
def two_court_layout() -> MultiCourtLayout:
    """Return two translated accepted courts with reciprocal transforms."""
    courts = []
    for index, translation_x in enumerate((0.0, 30.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = translation_x
        scene_from_court = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"error": 0.1},
                holdout_status="accepted",
                holdout_metrics={"error": 0.2},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -30.0, -2.0, 50.0, 30.0, 20.0),
        primary_court_instance_id="court-0",
    )


@pytest.fixture
def default_camera_profile() -> CameraProfileConfig:
    """Return a small six-camera profile whose values all come from config."""
    slots = []
    positions = (
        (-8.0, -12.0, 5.0),
        (8.0, -12.0, 5.0),
        (-8.0, 12.0, 5.0),
        (8.0, 12.0, 5.0),
        (0.0, -14.0, 7.0),
        (0.0, 14.0, 7.0),
    )
    for index, (x, y, z) in enumerate(positions):
        slots.append(
            {
                "slot_id": f"slot-{index}",
                "position_x_m": [x, x],
                "position_y_m": [y, y],
                "height_m": [z, z],
                "look_at_x_m": [0.0, 0.0],
                "look_at_y_m": [0.0, 0.0],
                "look_at_height_m": [0.5, 0.5],
                "hfov_degrees": [60.0, 60.0],
            }
        )
    return CameraProfileConfig.from_mapping(
        {
            "profile": "default",
            "image_size": [32, 24],
            "expected_camera_count": 6,
            "slots": slots,
        }
    )


def make_trajectory(
    trajectory_id: str,
    *,
    frame_count: int = 5,
    split: str = "train",
) -> BLCSTrajectory:
    """Build one smooth, fully present source trajectory."""
    positions: NDArray[np.float64] = np.zeros((frame_count, 1, 3), dtype=np.float64)
    positions[:, 0, 0] = np.linspace(-1.0, 1.0, frame_count)
    positions[:, 0, 2] = 1.5
    velocities = (
        np.gradient(positions, axis=0)
        if frame_count > 1
        else np.zeros_like(positions)
    )
    present: NDArray[np.bool_] = np.ones((frame_count, 1), dtype=np.bool_)
    return BLCSTrajectory(
        trajectory_id=trajectory_id,
        split=split,
        fps=30.0,
        positions_court_m=positions,
        velocities_court_mps=velocities,
        present=present,
        tracks=(
            BLCSTrack(
                object_id="ball-001",
                source_trajectory_id=trajectory_id,
                source_frame_indices=tuple(range(frame_count)),
            ),
        ),
        source_metadata={"physics": "test"},
    )


@pytest.fixture
def blcs_trajectory_factory():
    """Return the focused trajectory constructor."""
    return make_trajectory
