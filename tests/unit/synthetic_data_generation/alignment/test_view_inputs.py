"""Tests for the current alignment view-input helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.synthetic_data_generation.alignment.view_inputs import (
    load_provider_rgb_image,
    partition_fit_and_holdout_cameras,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def _camera(camera_id: str, group_id: int) -> SceneCamera:
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="provider-camera",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=group_id,
        group_id=group_id,
        width=4,
        height=3,
        intrinsics=(3.0, 0.0, 2.0, 0.0, 3.0, 1.5, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in np.eye(4).ravel()),
    )


def test_partition_uses_group_ids_and_preserves_camera_order() -> None:
    cameras = (
        _camera("camera-0", 0),
        _camera("camera-2", 2),
        _camera("camera-1", 1),
    )

    fit, holdout = partition_fit_and_holdout_cameras(
        cameras,
        holdout_group_ids=(1,),
    )

    assert tuple(camera.camera_id for camera in fit) == ("camera-0", "camera-2")
    assert tuple(camera.camera_id for camera in holdout) == ("camera-1",)


def test_load_provider_rgb_image_returns_uint8_three_channel_rgb(
    tmp_path: Path,
) -> None:
    rgba = np.asarray(
        [[[10, 20, 30, 0], [40, 50, 60, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "provider.png"
    Image.fromarray(rgba).save(path)

    image = load_provider_rgb_image(path)

    assert image.dtype == np.uint8
    assert image.shape == (1, 2, 3)
    np.testing.assert_array_equal(image, rgba[..., :3])
