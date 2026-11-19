"""Independent known-H raster paint, net tape and occlusion contracts."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.tasks.court_detection.geometry.homography import (
    COURT_HOMOGRAPHY_EDGES,
    court_template_xy,
)
from src.tasks.court_detection.geometry.paint_observations import PaintConfig
from src.tasks.court_detection.geometry.paint_refinement import refine_paint


def painted_court(*, occluded=False, partial=False, lines=range(9), net=False):
    matrix = np.array([[28., 3., 320.], [1., 14., 240.], [.002, .001, 1.]])
    if partial:
        matrix[0, 2] -= 180
    points = cv2.perspectiveTransform(court_template_xy()[None].astype(float), matrix)[0]
    high: np.ndarray = np.full((1440, 1920, 3), [35, 85, 110], np.uint8)
    for index in lines:
        a, b = COURT_HOMOGRAPHY_EDGES[index]
        cv2.line(high, tuple(np.rint(points[a] * 3).astype(int)), tuple(np.rint(points[b] * 3).astype(int)), (235, 235, 235), 9, cv2.LINE_AA)
    if net:
        cv2.line(high, (100, 680), (1820, 700), (255, 255, 255), 10, cv2.LINE_AA)
    if occluded:
        cv2.rectangle(high, (900, 450), (1040, 990), (20, 25, 30), -1)
    return cv2.resize(high, (640, 480), interpolation=cv2.INTER_AREA), matrix, points


@pytest.mark.parametrize("occluded,partial", [(False, False), (True, False), (False, True)])
def test_rgb_centres_recover_known_h_with_net_and_occlusion(occluded, partial):
    image, matrix, truth = painted_court(occluded=occluded, partial=partial, net=True)
    translation = np.array([[1., 0, 3.], [0, 1., -2.], [0, 0, 1.]])
    result = refine_paint(image, translation @ matrix)
    assert result.quality.accepted, result.quality.diagnostics()
    visible = (truth[:, 0] > 0) & (truth[:, 0] < 639)
    assert np.max(np.linalg.norm(result.projected[visible] - truth[visible], axis=1)) < 1.5
    assert result.quality.metrics["lost_initial_lines"] == []
    assert result.quality.metrics["schema"] == "court_rgb_quality_v1"


def test_fit_budget_failure_and_absent_paint_do_not_return_initial_h():
    image, matrix, _ = painted_court()
    shifted = matrix.copy()
    shifted[0, 2] += 8
    result = refine_paint(image, shifted, replace(PaintConfig(), max_nfev=1))
    assert result.matrix is None and np.isnan(result.projected).all()
    assert result.quality.reasons == ("paint_optimization_failed",)
    np.testing.assert_array_equal(result.initial_matrix, shifted)
    blank = refine_paint(np.full_like(image, 80), matrix)
    assert blank.matrix is None and "insufficient_paint_distribution" in blank.quality.reasons


def test_successful_fit_with_uncertain_projection_is_not_accepted():
    image, matrix, _ = painted_court(partial=True)
    result = refine_paint(image, matrix, replace(PaintConfig(), max_projection_std_px=.01))
    assert result.fitted_matrix is not None
    assert result.matrix is None and "uncertain_paint_extrapolation" in result.quality.reasons
    assert np.isnan(result.projected).all()


def test_unconfirmed_net_tape_is_diagnostic_not_required_paint():
    image, matrix, _ = painted_court(lines=[0, 1, 2, 3, 4, 5, 7, 8])
    points = cv2.perspectiveTransform(court_template_xy()[None].astype(float), matrix)[0]
    # Replace one service line by a parallel net-like stripe displaced 15 px.
    a, b = points[[8, 9]] + [0, 15]
    cv2.line(image, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
    result = refine_paint(image, matrix)
    assert result.quality.accepted, result.quality.diagnostics()
    assert np.max(np.linalg.norm(result.projected - points, axis=1)) < 1.5
    assert "far_service" in result.quality.metrics["initial_stripe_lines"]
    assert "far_service" not in result.quality.metrics["initial_supported_lines"]
    assert "far_service" in result.quality.metrics["lost_unconfirmed_stripes"]


def test_confirmed_paint_line_loss_rejects_an_otherwise_solvable_fit(monkeypatch):
    import src.tasks.court_detection.geometry.paint_refinement as module

    image, matrix, _ = painted_court()
    extract = module.extract_paint_lines
    calls = 0

    def lose_baseline(*args, **kwargs):
        nonlocal calls
        calls += 1
        lines = extract(*args, **kwargs)
        # Coarse, model-local verification, then final extraction. Simulate an
        # observation lost at that boundary while eight other lines still fit.
        if calls == 3:
            return (replace(lines[0], status="insufficient_coverage"), *lines[1:])
        return lines

    monkeypatch.setattr(module, "extract_paint_lines", lose_baseline)
    result = refine_paint(image, matrix)
    assert result.fitted_matrix is not None
    assert result.matrix is None and np.isnan(result.projected).all()
    assert "lost_initial_paint_lines" in result.quality.reasons
    assert "far_baseline" in result.quality.metrics["initial_supported_lines"]
    assert result.quality.metrics["lost_initial_lines"] == ["far_baseline"]


def test_essential_verified_lines_do_not_require_an_unidentifiable_leave_out_fit():
    # Two baselines and three sidelines determine the projective grid. Dropping
    # either baseline loses rank; this is missing redundancy, not a 300 px error.
    image, matrix, truth = painted_court(lines=[0, 1, 2, 3, 4])
    result = refine_paint(image, matrix)
    assert result.quality.accepted, result.quality.diagnostics()
    assert np.max(np.linalg.norm(result.projected - truth, axis=1)) < 1.5
    stability = result.quality.metrics["stability"]
    assert stability["essential_lines"] == ["far_baseline", "near_baseline"]
    assert "far_baseline" not in stability["leave_line_out_max_px"]
    assert result.quality.metrics["unconfirmed_essential_lines"] == []

    # The same number of lines clustered at the near end cannot certify the
    # visible far corners. The intact covariance must still reject that case.
    image, matrix, _ = painted_court(lines=[1, 2, 3, 4, 7])
    uncertain = refine_paint(image, matrix)
    assert uncertain.matrix is None
    assert "uncertain_paint_extrapolation" in uncertain.quality.reasons


def test_coarse_primary_stripe_can_remain_a_semantic_alternative():
    # With only two transverse markings, an alternative baseline can define a
    # different projective court. A model-local match alone cannot disambiguate.
    image, matrix, truth = painted_court(lines=[0, 1, 2, 3, 4])
    a, b = truth[[0, 1]] + [0, 10]
    cv2.line(image, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
    result = refine_paint(image, matrix)
    assert result.fitted_matrix is not None
    assert result.matrix is None
    assert "ambiguous_paint_correspondence" in result.quality.reasons
    assert "far_baseline" in result.quality.metrics["unresolved_semantic_alternatives"]
