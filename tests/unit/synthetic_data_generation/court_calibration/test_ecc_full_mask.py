"""Explicit all-valid ECC masks preserve the unmasked registration result."""

import cv2
import numpy as np
import pytest

from src.synthetic_data_generation.court_calibration.matching import (
    distance_image,
    refine,
)


@pytest.mark.parametrize("translation", [0.0, 2.0, -1.5])
def test_full_input_mask_matches_unmasked_ecc(translation: float) -> None:
    template: np.ndarray = np.zeros((144, 256), dtype=np.uint8)
    cv2.rectangle(template, (35, 20), (220, 125), 255, 2)
    cv2.line(template, (35, 75), (220, 75), 255, 2)
    cv2.line(template, (120, 20), (120, 75), 255, 2)
    transform = np.array(
        [[1.0, 0.002, translation], [0.001, 1.0, translation / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    query = cv2.warpPerspective(template, transform, (256, 144), flags=cv2.INTER_NEAREST)
    # This overload uses no mask and OpenCV's default Gaussian filter size of 5.
    expected_score, expected_warp = cv2.findTransformECC(
        distance_image(query, 12.0),
        distance_image(template, 12.0),
        np.eye(3, dtype=np.float32),
        cv2.MOTION_HOMOGRAPHY,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-6),
    )
    homography, warp, score, status = refine(template, query, np.eye(3))
    assert status == "ecc_returned_estimate_not_geometric_verification"
    assert homography is not None and warp is not None and score is not None
    assert score == pytest.approx(expected_score, abs=1e-10)
    np.testing.assert_allclose(warp, expected_warp, atol=1e-10, rtol=0)
    expected_homography = np.linalg.inv(expected_warp.astype(np.float64))
    expected_homography /= expected_homography[2, 2]
    np.testing.assert_allclose(homography, expected_homography, atol=1e-10, rtol=0)
