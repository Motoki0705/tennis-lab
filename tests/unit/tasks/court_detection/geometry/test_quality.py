"""Known court markings, image clipping and unsupported extrapolation."""

import cv2
import numpy as np

from src.tasks.court_detection.geometry.quality import (
    CourtQualityConfig,
    assess_court_quality,
    projection_jacobian,
)
from tests.unit.tasks.court_detection.geometry.test_hybrid_homography import (
    EDGES,
    scene,
)


def test_projection_jacobian_has_expected_eight_constraints() -> None:
    result = projection_jacobian(np.array([[2., 3.]]), np.eye(3))
    np.testing.assert_array_equal(result[0], [[2, 3, 1, 0, 0, 0, -4, -6], [0, 0, 0, 2, 3, 1, -6, -9]])


def test_visible_partial_court_is_accepted_without_inventing_off_image_observations() -> None:
    template, truth, _ = scene()
    truth -= [40, 0]
    matrix, _ = cv2.findHomography(template, truth, method=0)
    probability: np.ndarray = np.zeros((320, 320), np.float32)
    for a, b in EDGES:
        cv2.line(probability, tuple(np.rint(truth[a]).astype(int)), tuple(np.rint(truth[b]).astype(int)), .95, 3)
    valid = (truth[:, 0] >= 0) & (truth[:, 0] < 320)
    assert not valid.all() and valid.sum() >= 4
    raw = truth.copy()
    raw[~valid] = np.nan
    quality = assess_court_quality(matrix, raw, valid, probability, image_size_hw=(320, 320),
                                  template_xy=template, config=CourtQualityConfig())
    assert quality.accepted, quality.diagnostics()
    assert quality.score is not None
    assert quality.metrics["stability_status"] == "ok"
    assert quality.diagnostics()["schema"] == "court_quality_v1"


def test_one_line_cannot_certify_a_homography_even_with_exact_raw_points() -> None:
    template, truth, _ = scene()
    matrix, _ = cv2.findHomography(template, truth, method=0)
    probability: np.ndarray = np.zeros((320, 320), np.float32)
    cv2.line(probability, tuple(np.rint(truth[0]).astype(int)), tuple(np.rint(truth[1]).astype(int)), .95, 3)
    quality = assess_court_quality(matrix, truth, np.ones(14, bool), probability,
        image_size_hw=(320, 320), template_xy=template, config=CourtQualityConfig())
    assert not quality.accepted
    assert "insufficient_line_distribution" in quality.reasons


def test_wrong_visible_marking_is_not_hidden_by_an_average_support() -> None:
    template, truth, probability = scene()
    matrix, _ = cv2.findHomography(template, truth, method=0)
    # Remove the near baseline and draw an incompatible parallel marking.
    cv2.line(probability, tuple(np.rint(truth[2]).astype(int)), tuple(np.rint(truth[3]).astype(int)), 0., 11)
    quality = assess_court_quality(matrix, truth, np.ones(14, bool), probability,
        image_size_hw=(320, 320), template_xy=template, config=CourtQualityConfig())
    assert not quality.accepted
    assert "visible_line_mismatch" in quality.reasons
    line = quality.metrics["line"]
    assert isinstance(line, dict)
    assert line["forward_support"] > .55
