from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization.reveal.compositor import (
    gaussian_layer,
    over,
    to_uint8,
)
from src.synthetic_data_generation.visualization.reveal.contracts import RevealPhase
from src.synthetic_data_generation.visualization.reveal.layers import (
    draw_cameras,
    draw_points,
    project_cameras,
    project_points,
)
from src.synthetic_data_generation.visualization.reveal.sources import (
    crop_points_to_court_box,
    radial_reveal_order,
)
from src.synthetic_data_generation.visualization.reveal.timeline import (
    NHTAppearanceState,
    RevealFrameState,
)
from src.synthetic_data_generation.visualization.reveal.viewpoint import RevealCamera


def _camera() -> RevealCamera:
    """Identity pose looking along +z with a 20 x 10 image."""
    return RevealCamera(
        width=20,
        height=10,
        intrinsics=np.asarray(((10.0, 0.0, 9.5), (0.0, 10.0, 4.5), (0.0, 0.0, 1.0))),
        camera_to_scene=np.eye(4),
    )


def _state(**overrides: float) -> RevealFrameState:
    values = {
        "camera_fraction": 1.0,
        "camera_opacity": 0.0,
        "sfm_fraction": 1.0,
        "sfm_opacity": 0.0,
        "gaussian_opacity": 1.0,
        "color_progress": 1.0,
        "matte_release": 1.0,
    }
    values.update(overrides)
    return RevealFrameState(
        phase=RevealPhase.HOLD,
        nht=NHTAppearanceState(splat_progress=1.0, view_dependence=1.0),
        **values,
    )


def test_point_splats_keep_the_nearest_point_per_pixel() -> None:
    # Both points land on pixel (9, 4); the second is nearer and red.
    points: NDArray[Any] = np.asarray(
        [[0.0, 0.0, 4.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    projected = project_points(_camera(), points, np.asarray([0, 1]), radius_px=1)

    rgb, alpha = draw_points(projected, 1.0)

    np.testing.assert_allclose(rgb[4, 9], (1.0, 0.0, 0.0))
    assert alpha.sum() == 1.0


def test_point_reveal_draws_a_prefix_of_the_reveal_order() -> None:
    points: NDArray[Any] = np.asarray(
        [[-0.5, 0.0, 2.0, 1.0, 1.0, 1.0], [0.5, 0.0, 2.0, 1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    projected = project_points(_camera(), points, np.asarray([1, 0]), radius_px=2)

    _, half = draw_points(projected, 0.5)
    _, none = draw_points(projected, 0.0)

    assert none.sum() == 0.0
    # Only point 1 (first in reveal order) is drawn as a 3 x 3 splat.
    assert half.sum() == 9.0
    # Point 1 projects to column 12, so its splat covers columns 11..13.
    assert half[:, 11:14].sum() == 9.0


def test_points_behind_the_camera_are_dropped() -> None:
    points: NDArray[Any] = np.asarray([[0.0, 0.0, -2.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    projected = project_points(_camera(), points, np.asarray([0]), radius_px=1)

    assert len(projected.pixel_index) == 0


def test_reveal_order_must_be_a_permutation() -> None:
    points: NDArray[Any] = np.zeros((2, 6), dtype=np.float32)
    points[:, 2] = 1.0
    with pytest.raises(ValueError, match="permutation"):
        project_points(_camera(), points, np.asarray([0, 0]), radius_px=1)


def test_camera_layer_reveals_cameras_in_capture_order() -> None:
    capture = tuple(
        SceneCamera(
            camera_id=f"frame-{index}",
            source_frame_index=index,
            width=4,
            height=4,
            intrinsics=(2.0, 0.0, 1.5, 0.0, 2.0, 1.5, 0.0, 0.0, 1.0),
            camera_to_scene=RigidTransform.from_matrix(
                np.asarray(
                    ((1, 0, 0, x), (0, 1, 0, 0), (0, 0, 1, 5.0), (0, 0, 0, 1)),
                    dtype=np.float64,
                )
            ),
            image_path=f"images/{index}.jpg",
        )
        for index, x in enumerate((-2.0, 0.0, 2.0))
    )
    projected = project_cameras(_camera(), capture, frustum_depth_scene=0.5, frustum_stride=2)

    assert projected.frustum_camera_index.tolist() == [0, 2]
    _, nothing = draw_cameras(projected, 0.0, rgb=(1.0, 0.5, 0.0), line_width_px=1)
    rgb, some = draw_cameras(projected, 2 / 3, rgb=(1.0, 0.5, 0.0), line_width_px=1)
    _, everything = draw_cameras(projected, 1.0, rgb=(1.0, 0.5, 0.0), line_width_px=1)

    assert nothing.sum() == 0.0
    assert 0.0 < some.sum() < everything.sum()
    np.testing.assert_allclose(rgb[..., 1], some[..., 0] * 0.5, atol=1e-6)


def test_final_gaussian_layer_is_the_raw_nht_rgb() -> None:
    rgb: NDArray[Any] = np.full((2, 2, 3), 0.25, dtype=np.float32)
    alpha: NDArray[Any] = np.asarray([[[0.0], [0.5]], [[1.0], [1.0]]], dtype=np.float32)
    background: NDArray[Any] = np.ones((2, 2, 3), dtype=np.float32)

    layer_rgb, layer_alpha = gaussian_layer(_state(), rgb, alpha, (1.0, 0.0, 0.0))
    frame = over(background, layer_rgb, layer_alpha, 1.0)

    np.testing.assert_allclose(frame, rgb)


def test_uniform_points_are_matted_over_the_background() -> None:
    rgb: NDArray[Any] = np.full((1, 2, 3), 0.25, dtype=np.float32)
    alpha: NDArray[Any] = np.asarray([[[0.0], [1.0]]], dtype=np.float32)
    background: NDArray[Any] = np.zeros((1, 2, 3), dtype=np.float32)

    layer = gaussian_layer(
        _state(color_progress=0.0, matte_release=0.0), rgb, alpha, (1.0, 0.0, 0.0)
    )
    frame = over(background, *layer, 1.0)

    np.testing.assert_allclose(frame[0, 0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(frame[0, 1], (1.0, 0.0, 0.0))


def test_layer_opacity_fades_and_quantization_clips() -> None:
    base: NDArray[Any] = np.zeros((1, 1, 3), dtype=np.float32)
    layer: NDArray[Any] = np.ones((1, 1, 3), dtype=np.float32)
    alpha: NDArray[Any] = np.ones((1, 1, 1), dtype=np.float32)

    np.testing.assert_allclose(over(base, layer, alpha, 0.25), 0.25)
    assert over(base, layer, alpha, 0.0) is base
    assert to_uint8(np.asarray([[[-1.0, 0.5, 2.0]]], dtype=np.float32)).tolist() == [
        [[0, 128, 255]]
    ]


def test_crop_keeps_points_inside_the_court_frame_box() -> None:
    scene_from_court = np.eye(4)
    scene_from_court[:3, 3] = (10.0, 0.0, 0.0)
    points: NDArray[Any] = np.asarray(
        [[10.0, 0.0, 0.0, 1, 1, 1], [20.0, 0.0, 0.0, 1, 1, 1]], dtype=np.float32
    )

    cropped = crop_points_to_court_box(
        points, scene_from_court, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
    )

    np.testing.assert_array_equal(cropped, points[:1])
    with pytest.raises(ValueError, match="no points"):
        crop_points_to_court_box(
            points, scene_from_court, (5.0, 6.0, -1.0, 1.0, -1.0, 1.0)
        )


def test_radial_order_starts_nearest_the_centre() -> None:
    points: NDArray[Any] = np.asarray(
        [[3.0, 0, 0, 1, 1, 1], [1.0, 0, 0, 1, 1, 1], [2.0, 0, 0, 1, 1, 1]],
        dtype=np.float32,
    )

    assert radial_reveal_order(points, np.zeros(3)).tolist() == [1, 2, 0]
