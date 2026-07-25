"""Tests for projected court line evidence."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.evaluation.image_evidence import (
    COURT_LINE_EDGES,
    line_edge_support,
)
from src.tasks.court_detection.geometry import court_template_xy, project_points


def test_line_edge_support_is_high_for_matching_rendered_lines() -> None:
    homography = np.asarray(
        [
            [0.040, 0.002, 0.50],
            [0.002, -0.025, 0.48],
            [0.002, -0.012, 1.00],
        ],
        dtype=np.float32,
    )
    projected = project_points(court_template_xy(), homography)
    height, width = 720, 1280
    pixels = projected * np.asarray([width - 1, height - 1], dtype=np.float32)
    image: NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
    for first, second in COURT_LINE_EDGES:
        cv2.line(
            image,
            tuple(np.rint(pixels[first]).astype(int)),
            tuple(np.rint(pixels[second]).astype(int)),
            255,
            4,
        )

    matching = line_edge_support(
        image,
        projected,
        distance_tolerance_px=3.0,
        max_side=900,
    )
    shifted = line_edge_support(
        image,
        projected + np.asarray([0.0, 0.10], dtype=np.float32),
        distance_tolerance_px=3.0,
        max_side=900,
    )

    assert matching > 0.9
    assert shifted < 0.2
