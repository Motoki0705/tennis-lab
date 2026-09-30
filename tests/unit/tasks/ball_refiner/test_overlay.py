"""Geometry regressions for uncertainty overlays, independent of video codecs."""

import numpy as np
import pytest

from src.tasks.ball_refiner.visualization.overlay import component_geometry


def test_ellipses_preserve_source_covariance_and_component_weights() -> None:
    means = np.array([[.1, .7], [.9, .2]])
    factors = np.array([[[.02, 0], [.01, .03]], [[.08, 0], [-.03, .04]]])
    logits = np.log(np.array([.25, .75]))
    # Anisotropic source mapping exposes swapping row/column scaling.
    scale = (959.5, 539.5)
    mean, ellipses = component_geometry(means, factors, logits, scale)
    np.testing.assert_allclose(mean, (.25 * means[0] + .75 * means[1]) * scale)
    for i, ellipse in enumerate(ellipses):
        angle = np.radians(ellipse.angle_degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        recovered = rotation @ np.diag((np.array(ellipse.semiaxes) / 2) ** 2) @ rotation.T
        expected = np.diag(scale) @ factors[i] @ factors[i].T @ np.diag(scale)
        np.testing.assert_allclose(recovered, expected)
        np.testing.assert_allclose(ellipse.center, means[i] * scale)
        assert ellipse.weight == pytest.approx([.25, .75][i])
    # Widely separated components must not inflate the individual ellipses.
    shifted = means.copy()
    shifted[1] += 100
    _, after = component_geometry(shifted, factors, logits, scale)
    assert after[0].semiaxes == ellipses[0].semiaxes


@pytest.mark.parametrize('bad', [np.array([[[0., 0], [0, .1]]]), np.array([[[.1, .1], [0, .1]]]),
                                 np.array([[[np.nan, 0], [0, .1]]])])
def test_invalid_covariance_is_rejected_without_a_visual_fallback(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        component_geometry(np.zeros((1, 2)), bad, np.zeros(1), (100, 100))
