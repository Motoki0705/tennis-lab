from __future__ import annotations

import numpy as np

from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_distance,
    ground_footpoints,
)
from src.utils.geometry.triangulation import PinholeCamera


def _camera() -> PinholeCamera:
    center = np.array([0.0, -20.0, 6.0])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera("c", np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]]), rotation, -rotation @ center)


def test_box_bottom_centre_lands_on_the_ground_point_and_truncated_boxes_are_invalid() -> None:
    camera = _camera()
    feet = np.array([[1.0, 2.0, 0.0], [-3.0, 5.0, 0.0]])
    uv = camera.project(feet)[0]
    boxes = np.array([[u - 20, v - 120, u + 20, v] for u, v in uv])
    points, valid = ground_footpoints(boxes, np.array([True, True]), camera, 1080, FootpointConfig())
    assert valid.all()
    np.testing.assert_allclose(points, feet[:, :2], atol=1e-6)
    boxes[1, 3] = 1079.0  # bottom at the image border: feet cut off
    _, valid = ground_footpoints(boxes, np.array([True, True]), camera, 1080, FootpointConfig(bottom_border_px=4))
    assert valid.tolist() == [True, False]
    _, valid = ground_footpoints(boxes, np.array([False, True]), camera, 1080, FootpointConfig(bottom_border_px=0))
    assert valid.tolist() == [False, True]


def test_ground_distance_uses_only_shared_valid_frames() -> None:
    a = np.array([[0.0, 0], [0, 0], [0, 0], [9, 9]])
    b = np.array([[1.0, 0], [3, 0], [5, 0], [0, 0]])
    result = ground_distance(a, np.array([True, True, True, False]), b, np.array([True, True, True, True]))
    assert result.shared_frames == 3 and result.median_m == 3.0
    empty = ground_distance(a, np.zeros(4, bool), b, np.ones(4, bool))
    assert empty.shared_frames == 0 and np.isnan(empty.median_m)
