"""Tests for the court footpoint ROI polygon."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.utils.geometry.court_roi import court_footpoint_polygon_px
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    CourtConfig,
    court_keypoints_3d,
)


def test_court_footpoint_polygon_projects_explicit_physical_margins() -> None:
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    polygon = court_footpoint_polygon_px(
        physical * 10.0 + 200.0, size=(400, 400), sideline_margin_m=1.0, baseline_margin_m=2.0,
    )
    expected = np.asarray([
        (-HALF_DOUBLES_WIDTH - 1.0, HALF_LENGTH + 2.0),
        (HALF_DOUBLES_WIDTH + 1.0, HALF_LENGTH + 2.0),
        (HALF_DOUBLES_WIDTH + 1.0, -HALF_LENGTH - 2.0),
        (-HALF_DOUBLES_WIDTH - 1.0, -HALF_LENGTH - 2.0),
    ]) * 10.0 + 200.0
    actual = np.asarray(polygon)
    distances = np.linalg.norm(expected[:, None] - actual[None], axis=-1)
    assert actual.shape == (4, 2)
    assert len(set(distances.argmin(axis=1))) == 4
    assert float(distances.min(axis=1).max()) < 1e-4


@pytest.mark.parametrize("baseline_margin", [1.0, 30.0])
@pytest.mark.parametrize("half_turn", [False, True])
def test_court_footpoint_region_matches_world_bounds_even_past_camera(baseline_margin: float, half_turn: bool) -> None:
    # The camera plane is world y=-33.33. A 30m margin crosses it, although
    # the court itself remains entirely visible in this perspective image.
    homography: np.ndarray = np.array([[50.0, 10.0, 640.0], [0.0, -5.0, 360.0], [0.0, 0.03, 1.0]])
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    projected = cv2.perspectiveTransform(physical[None], homography)[0]
    view = build_court_view_record(
        camera_id="cam0", camera_center_court_m=[0.0, 12.0 if half_turn else -12.0, 5.0],
        contract=resolve_court_keypoint_contract("camera_view_v2"),
    )
    keypoints = projected[np.asarray(view.semantic_to_physical[:14])]
    polygon = np.asarray(court_footpoint_polygon_px(
        keypoints, size=(1280, 720), sideline_margin_m=1.0, baseline_margin_m=baseline_margin,
    ), dtype=np.float32)
    assert np.isfinite(polygon).all()
    assert (polygon >= -1e-6).all() and (polygon <= [1280 + 1e-6, 720 + 1e-6]).all()
    assert cv2.isContourConvex(polygon)
    assert cv2.pointPolygonTest(polygon, (640.0, 360.0), False) >= 0

    # Independent membership oracle: map test pixels back to the court plane.
    pixels = np.random.default_rng(0).uniform([0, 0], [1280, 720], size=(500, 2))
    homogeneous = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(homography).T
    world = homogeneous[:, :2] / homogeneous[:, 2:]
    expected = (
        (homogeneous[:, 2] > 0)
        & (abs(world[:, 0]) <= HALF_DOUBLES_WIDTH + 1.0)
        & (abs(world[:, 1]) <= HALF_LENGTH + baseline_margin)
    )
    actual = np.array([cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0 for x, y in pixels])
    np.testing.assert_array_equal(actual, expected)


def test_invalid_margins_are_rejected() -> None:
    physical = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    with pytest.raises(ValueError, match="margins"):
        court_footpoint_polygon_px(physical * 10 + 200, size=(400, 400), sideline_margin_m=-1.0, baseline_margin_m=0.0)
