"""Tests for seven-class, instance-aware multi-peak court labels."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.court.labels import (
    PHYSICAL_TO_SYMMETRIC_CLASS,
    build_seven_channel_heatmaps,
    decode_heatmap_atlas_u16,
    encode_heatmap_atlas_u16,
    project_multi_court,
    rescale_projection,
)
from src.synthetic_data_generation.court.layout import MultiCourtLayout
from src.synthetic_data_generation.scene_contract import SceneCamera
from tests.unit.synthetic_data_generation.court.conftest import make_camera


def test_physical_line_points_form_seven_unordered_pairs() -> None:
    counts = np.bincount(PHYSICAL_TO_SYMMETRIC_CLASS, minlength=7)
    np.testing.assert_array_equal(counts, np.full(7, 2))


def test_two_courts_merge_four_peaks_per_class_without_instance_channels(
    two_court_layout: MultiCourtLayout,
) -> None:
    camera = make_camera("camera_0", (7.5, -42.0, 22.0))
    projection = project_multi_court(camera, two_court_layout)
    assert len(projection.courts) == 2
    assert all(len(value.points) == 2 for court in projection.courts for value in court.classes)
    assert all(
        court.in_frame_point_count == 14 for court in projection.courts
    )

    heatmaps = build_seven_channel_heatmaps(
        projection,
        sigma_px=1.5,
        require_renderer_visibility=False,
    )

    assert heatmaps.shape == (7, camera.height, camera.width)
    for class_id in range(7):
        projected_points = [
            point
            for court in projection.courts
            for point in court.classes[class_id].points
        ]
        assert len(projected_points) == 4
        for point in projected_points:
            x = int(round(point.uv[0]))
            y = int(round(point.uv[1]))
            assert heatmaps[class_id, y, x] >= 0.85


def test_renderer_visibility_is_never_silently_invented(
    two_court_layout: MultiCourtLayout,
) -> None:
    camera = make_camera("camera_0", (7.5, -42.0, 22.0))
    projection = project_multi_court(camera, two_court_layout)

    with pytest.raises(ValueError, match="remains unevaluated"):
        build_seven_channel_heatmaps(
            projection,
            sigma_px=2.0,
            require_renderer_visibility=True,
        )


def test_partial_court_is_a_first_class_coverage_bucket(
    two_court_layout: MultiCourtLayout,
) -> None:
    camera: SceneCamera = make_camera(
        "camera_0",
        (7.5, -25.0, 3.0),
        target=(0.0, 5.0, 0.0),
    )
    projection = project_multi_court(camera, two_court_layout)

    assert any(
        court.coverage_bucket in {"partial", "near_full"}
        for court in projection.courts
    )


def test_rescale_projection_preserves_semantics(
    two_court_layout: MultiCourtLayout,
) -> None:
    camera = make_camera("camera_0", (7.5, -42.0, 22.0))
    projection = project_multi_court(camera, two_court_layout)
    resized = rescale_projection(projection, width=40, height=30)

    assert (resized.width, resized.height) == (40, 30)
    assert [court.court_instance_id for court in resized.courts] == [
        court.court_instance_id for court in projection.courts
    ]
    original = projection.courts[0].classes[0].points[0]
    scaled = resized.courts[0].classes[0].points[0]
    assert scaled.uv == pytest.approx(
        (
            original.uv[0] * 40 / projection.width,
            original.uv[1] * 30 / projection.height,
        )
    )
    assert scaled.xyz_scene == original.xyz_scene
    assert scaled.visible is original.visible


def test_heatmap_atlas_round_trip_is_bounded_and_seven_channel() -> None:
    heatmaps: NDArray[np.float32] = np.zeros(
        (7, 5, 6), dtype=np.float32
    )
    for channel in range(7):
        heatmaps[channel, channel % 5, channel % 6] = (channel + 1) / 7

    atlas = encode_heatmap_atlas_u16(heatmaps)
    decoded = decode_heatmap_atlas_u16(atlas)

    assert atlas.dtype == np.uint16
    assert atlas.shape == (5, 42)
    assert decoded.shape == heatmaps.shape
    np.testing.assert_allclose(decoded, heatmaps, atol=1.0 / 65535.0, rtol=0.0)


@pytest.mark.parametrize(
    "heatmaps",
    (
        np.zeros((6, 5, 6), dtype=np.float32),
        np.full((7, 5, 6), np.nan, dtype=np.float32),
        np.full((7, 5, 6), 1.1, dtype=np.float32),
    ),
)
def test_heatmap_atlas_rejects_invalid_targets(
    heatmaps: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        encode_heatmap_atlas_u16(heatmaps)
