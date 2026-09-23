"""Synthetic ground-truth checks for the geometric initializer."""

import numpy as np
from numpy.testing import assert_allclose

from src.utils.geometry.triangulation import project_multiview, triangulate_multiview


def fixture_scene():
    k = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    p = np.stack(
        [
            k @ np.column_stack((np.eye(3), -np.array(c)))
            for c in [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        ]
    )
    x = np.random.default_rng(19).uniform(
        [-1.0, -1.0, 4.0], [1.0, 1.0, 8.0], (2, 17, 3)
    )
    uv, _ = project_multiview(x, p)
    return x, uv, p


def test_recovers_world_coordinates_with_missing_and_corrupt_view():
    x, uv, p = fixture_scene()
    score = np.ones(uv.shape[:-1])
    uv[0, 0, 0] = np.nan
    uv[0, 1, 1] += 200
    score[0, 1, 1] = 0.01
    result = triangulate_multiview(uv, score, p)
    assert result.valid.all()
    assert_allclose(result.points, x, atol=1e-8)
    assert not result.used_views[0, 0, 0]
    assert not result.used_views[0, 1, 1]


def test_refinement_reduces_weighted_pixel_error():
    _, uv, p = fixture_scene()
    uv += np.random.default_rng(31).normal(0, 5, uv.shape)
    score = np.random.default_rng(12).uniform(0.4, 1, uv.shape[:-1])
    initial = triangulate_multiview(uv, score, p, refinement_steps=0)
    refined = triangulate_multiview(uv, score, p)
    before = np.sum((initial.reprojected - uv) ** 2 * score[..., None], axis=(-1, -2))
    after = np.sum((refined.reprojected - uv) ** 2 * score[..., None], axis=(-1, -2))
    assert np.all(after <= before + 1e-7)
    assert np.any(after < before - 1e-5)


def test_rejects_missing_degenerate_and_behind_camera():
    x, uv, p = fixture_scene()
    score = np.ones(uv.shape[:-1])
    score[0, 0, :2] = 0
    result = triangulate_multiview(uv, score, p)
    assert not result.valid[0, 0]
    assert np.isnan(result.points[0, 0]).all()
    repeated = np.repeat(p[:1], 3, axis=0)
    repeated_uv, _ = project_multiview(x, repeated)
    assert not triangulate_multiview(
        repeated_uv, np.ones_like(score), repeated
    ).valid.any()
    x[..., 2] *= -1
    behind, _ = project_multiview(x, p)
    assert not triangulate_multiview(behind, np.ones_like(score), p).valid.any()


def test_missing_camera_on_projection_plane_does_not_poison_visible_pair():
    x = np.array([[0.0, 0.0, 6.0]])
    p = np.stack(
        [
            np.column_stack((np.eye(3), t))
            for t in [
                np.array([2.0, 0.0, 0.0]),
                np.array([-2.0, 0.0, 0.0]),
                np.array([0.0, -2.0, -6.0]),
            ]
        ]
    )
    uv, _ = project_multiview(x, p)
    score = np.array([[1.0, 1.0, 0.0]])
    result = triangulate_multiview(uv, score, p)
    assert result.valid.all()
    assert_allclose(result.points, x, atol=1e-8)
