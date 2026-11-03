import numpy as np
import pytest

from src.utils.geometry.kalman import smooth_positions


def test_scalar_update_matches_independent_gaussian_conditioning():
    xyz = np.array([[0., 2, -3], [2, -1, 5]])
    cov = np.tile(np.eye(3), (2, 1, 1))
    mean, covariance = smooth_positions(xyz, cov, np.ones(2, bool), fps=1,
        acceleration_std=2, initial_velocity_std=1, backward=False)
    # Predicted position variance 1+1+2²/4=3; K=3/(3+1)=.75.
    np.testing.assert_allclose(mean, [xyz[0], xyz[0]+.75*(xyz[1]-xyz[0])])
    np.testing.assert_allclose(covariance[1], np.eye(3)*.75)


def test_rts_uses_fps_and_resets_across_missing_frame():
    xyz = np.array([[1., 2, 3], [1.2, 1.8, 3.1], [1.35, 1.65, 3.22],
                    [0, 0, 0], [60, -40, 20], [62, -38, 19]])
    valid = np.array([True, True, True, False, True, True])
    cov = np.tile(np.diag([.01, .02, .03]), (6, 1, 1))
    cov[3] = 0
    a, acov = smooth_positions(xyz, cov, valid, fps=10, acceleration_std=2, initial_velocity_std=4)
    changed = xyz.copy()
    changed[4:] *= 100
    b, _ = smooth_positions(changed, cov, valid, fps=10, acceleration_std=2, initial_velocity_std=4)
    np.testing.assert_array_equal(a[:4], b[:4])
    np.testing.assert_array_equal(a[3], [0, 0, 0])
    assert np.linalg.eigvalsh(acov[valid]).min() > 0
    assert np.trace(acov[0]) < np.trace(cov[0])
    fast, _ = smooth_positions(xyz, cov, valid, fps=100, acceleration_std=2, initial_velocity_std=4)
    assert np.linalg.norm(a[2]-fast[2]) > .001


def test_invalid_covariance_and_noise_fail():
    with pytest.raises(ValueError, match="positive definite"):
        smooth_positions(np.ones((2, 3)), np.zeros((2, 3, 3)), np.ones(2, bool), fps=10,
            acceleration_std=2, initial_velocity_std=4)
