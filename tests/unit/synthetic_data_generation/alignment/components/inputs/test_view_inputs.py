"""Tests for shared fit/holdout and RGB provider input boundaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.synthetic_data_generation.alignment.components.inputs.view_inputs import (
    load_provider_rgb_image,
    partition_fit_and_holdout_cameras,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def _camera(camera_id: str, group_id: int) -> SceneCamera:
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="synthetic",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=group_id,
        group_id=group_id,
        width=2,
        height=2,
        intrinsics=(1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in np.eye(4).ravel()),
    )


def test_partition_uses_group_ids_and_preserves_input_order() -> None:
    cameras = tuple(_camera(f"camera-{index}", index % 3) for index in range(6))

    fit, holdout = partition_fit_and_holdout_cameras(
        cameras,
        holdout_group_ids=(1,),
    )

    assert [camera.camera_id for camera in fit] == [
        "camera-0",
        "camera-2",
        "camera-3",
        "camera-5",
    ]
    assert [camera.camera_id for camera in holdout] == ["camera-1", "camera-4"]


def test_partition_allows_explicit_empty_side_for_stage_validation() -> None:
    cameras = (_camera("camera-0", 0),)

    fit, holdout = partition_fit_and_holdout_cameras(
        cameras,
        holdout_group_ids=(0,),
    )

    assert fit == ()
    assert holdout == cameras


def test_load_provider_rgb_image_converts_grayscale_to_uint8_rgb(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gray.png"
    Image.fromarray(
        np.asarray([[0, 255], [64, 128]], dtype=np.uint8),
        mode="L",
    ).save(path)

    loaded = load_provider_rgb_image(path)

    assert loaded.shape == (2, 2, 3)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded[..., 0], loaded[..., 1])
    np.testing.assert_array_equal(loaded[..., 1], loaded[..., 2])


def test_load_provider_rgb_image_discards_alpha_without_color_reordering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rgba.png"
    rgba = np.asarray([[[10, 20, 30, 0], [40, 50, 60, 255]]], dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(path)

    loaded = load_provider_rgb_image(path)

    np.testing.assert_array_equal(
        loaded,
        np.asarray([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8),
    )
