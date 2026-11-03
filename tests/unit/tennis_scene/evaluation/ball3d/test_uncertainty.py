from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import chi2

from src.tennis_scene.evaluation.ball3d.runner import run_method
from src.tennis_scene.evaluation.ball3d.uncertainty import (
    ellipse_vertices,
    project_covariance,
    score_uncertainty,
)
from tests.support.ball3d_evaluation import make_inputs, make_scoring_bundle
from tests.unit.tennis_scene.evaluation.ball3d.test_scoring import (
    FixedPointMethod,
    _labels,
)


def test_projected_covariance_matches_independent_quotient_rule_and_native_pixels():
    inputs = make_inputs()
    result = FixedPointMethod()(inputs)
    covariance = np.tile(np.diag([.04, .09, .01]), (4, 1, 1))
    covariance[~result.valid] = 0
    result = replace(result, covariance=covariance)
    got, valid = project_covariance(inputs, result, "right")
    # u=160+120*(x+2)/(8-z), v=120+130*(-4-y)/(8-z), at (1,2,1).
    jac = np.array([[120/7, 0, 360/49], [0, -130/7, -780/49]])
    np.testing.assert_allclose(got[0], jac@covariance[0]@jac.T, atol=1e-12)
    np.testing.assert_array_equal(valid, result.valid)
    np.testing.assert_array_equal(got[3], 0)
    with pytest.raises(ValueError, match="time offsets"):
        project_covariance(inputs, replace(result, time_offsets_seconds=np.array([.01, 0., 0.])), "right")


def test_ellipse_rotated_axes_have_exact_mahalanobis_radius():
    covariance = np.array([[9., 3.], [3., 4.]])
    center = np.array([53., 127.])
    points = ellipse_vertices(center, covariance)
    d = points-center
    np.testing.assert_allclose(np.einsum("ni,ij,nj->n", d, np.linalg.inv(covariance), d), chi2.ppf(.95, 2), atol=1e-13)


def test_coverage_uses_scored_camera_frame_cohort_and_approved_estimates():
    inputs = make_inputs()

    class Uncertain(FixedPointMethod):
        def __call__(self, selected):
            result = super().__call__(selected)
            covariance = np.tile(np.eye(3)*100, (4, 1, 1))
            covariance[~result.valid] = 0
            return replace(result, covariance=covariance)

    runs = run_method(Uncertain(), inputs, bundle_sha256="x", code_commit="y", code_fingerprint="z")
    report = score_uncertainty({"fixture": runs}, make_scoring_bundle(inputs), _labels(inputs), split_frame=2)
    metrics = report["methods"]["fixture"]["lovo"]
    assert metrics["all"]["all"]["count"] == 5
    assert metrics["tuning"]["estimated"]["count"] == 1
    assert metrics["evaluation"]["all"]["count"] == 2
    assert metrics["all"]["all"]["ellipse_coverage"]["0.95"] == 1
