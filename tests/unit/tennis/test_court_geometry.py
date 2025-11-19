from __future__ import annotations

import math

import torch

from src.tennis.geometry import court


def test_court_constants_match_spec() -> None:
    assert math.isclose(court.COURT_LENGTH, 23.77, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.HALF_LENGTH, 11.885, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.SINGLES_WIDTH, 8.23, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.HALF_SINGLES_WIDTH, 4.115, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.DOUBLES_WIDTH, 10.97, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.HALF_DOUBLES_WIDTH, 5.485, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.SERVICE_LINE_DISTANCE, 6.40, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.NET_HEIGHT_CENTER, 0.914, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(court.NET_HEIGHT_POST, 1.07, rel_tol=0, abs_tol=1e-6)


def test_court_keypoints_shape_and_values() -> None:
    pts = court.court_keypoints_3d()
    assert isinstance(pts, torch.Tensor)
    assert pts.shape == (20, 3)
    # spot check: index 14 is net center ground
    assert torch.allclose(pts[14], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    # spot check: index 19 is center strap top
    assert math.isclose(float(pts[19, 2]), court.NET_HEIGHT_CENTER, abs_tol=1e-6)


def test_sample_camera_on_fence_bounds() -> None:
    x, y, z = court.sample_camera_position_on_fence(0.0, "near")
    assert x == court.X_MIN and y == court.Y_MIN and math.isclose(z, 3.0)
    x, y, z = court.sample_camera_position_on_fence(1.0, "far")
    assert x == court.X_MAX and y == court.Y_MAX and math.isclose(z, 3.0)

