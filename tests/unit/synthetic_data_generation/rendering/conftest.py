"""Shared fixtures for synthetic-data rendering tests."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.scene_contract import SceneCamera


@pytest.fixture
def scene_camera() -> SceneCamera:
    """Small identity-pose OpenCV camera for renderer-port tests."""
    return SceneCamera(
        camera_id="camera-0",
        source_camera_id="source-0",
        image_uri="images/camera-0.png",
        source_frame_index=0,
        group_id=0,
        width=64,
        height=48,
        intrinsics=(
            50.0,
            0.0,
            31.5,
            0.0,
            50.0,
            23.5,
            0.0,
            0.0,
            1.0,
        ),
        camera_to_scene=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
