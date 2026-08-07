from __future__ import annotations

import random

import numpy as np

from src.tasks.ball_detection.data.components.augmentation import AffineAugmentation
from src.utils.geometry.affine import build_centered_affine_matrix, transform_points


def test_ball_affine_matches_shared_matrix_for_visible_coordinates() -> None:
    height, width = 80, 120
    frames: list[np.ndarray] = [
        np.zeros((height, width, 3), dtype=np.float32)
    ]
    coords = [[(70.0, 30.0), (0.0, 0.0)]]
    visibility = [[1.0, 0.0]]
    cfg = {
        "enabled": True,
        "prob": 1.0,
        "rotation_deg_range": (10.0, 10.0),
        "scale_range": (1.1, 1.1),
        "translate_x_ratio_range": (0.05, 0.05),
        "translate_y_ratio_range": (-0.025, -0.025),
        "shear_x_deg_range": (4.0, 4.0),
        "shear_y_deg_range": (-2.0, -2.0),
        "border_mode": "constant",
    }

    _, out_coords, out_visibility = AffineAugmentation(cfg).forward(
        frames,
        coords,
        visibility,
        rng=random.Random(123),
    )

    expected_matrix = build_centered_affine_matrix(
        width=width,
        height=height,
        rotation_degrees=10.0,
        translate=(width * 0.05, height * -0.025),
        scale=1.1,
        shear_degrees=(4.0, -2.0),
        center=((width - 1) / 2.0, (height - 1) / 2.0),
    )
    expected = transform_points(np.array([[70.0, 30.0]], dtype=np.float32), expected_matrix)[0]

    assert out_visibility == [[1.0, 0.0]]
    np.testing.assert_allclose(out_coords[0][0], expected, atol=1e-5)
    assert out_coords[0][1] == (0.0, 0.0)
