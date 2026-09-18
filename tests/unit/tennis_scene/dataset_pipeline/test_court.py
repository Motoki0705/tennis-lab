"""Static court fitting rejects insufficient evidence and resists outliers."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.court import (
    StaticCourtSettings,
    fit_static_court,
)
from src.utils.schema.court import CourtConfig, court_keypoints_3d


def test_static_fit_rejects_an_outlier_without_moving_the_court() -> None:
    world = court_keypoints_3d(CourtConfig(0.914, None)).numpy()[:14, :2]
    h = np.array([[35, 0, 960], [0, 20, 540], [0, 0.01, 1]], np.float64)
    expected = cv2.perspectiveTransform(world[None], h)[0]
    raw = np.repeat(expected[None], 5, axis=0)
    raw[:, 3] += [300, -100]
    scores: np.ndarray = np.ones((5, 14), np.float32)
    settings = StaticCourtSettings(Path("model.ckpt"), 5, 0.3, 6, 10, 5)
    fitted, _, receipt = fit_static_court(
        raw, scores, size=(1920, 1080), settings=settings
    )
    assert 3 not in receipt["inlier_channels"]
    np.testing.assert_allclose(fitted * [1920, 1080], expected, atol=1e-3)
    assert receipt["is_measured_calibration"] is False


def test_static_fit_refuses_unobserved_court_channels() -> None:
    settings = StaticCourtSettings(Path("model.ckpt"), 5, 0.3, 6, 10, 5)
    with pytest.raises(ValueError, match="temporal support"):
        fit_static_court(
            np.zeros((5, 14, 2)),
            np.zeros((5, 14)),
            size=(1920, 1080),
            settings=settings,
        )
