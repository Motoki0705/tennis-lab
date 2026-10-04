"""Distinguish seam impulses, constant bias, temporal noise and physical curves."""
import numpy as np
import pytest

from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import Array
from src.tasks.ball_refiner.refiner_3d.diffusion.roughness import (
    local_error_components,
    stencil_masks,
    window_owners,
)


def test_step_between_windows_is_counted_only_at_seams() -> None:
    owners = window_owners([{'owned_start': 0, 'owned_stop': 8}, {'owned_start': 8, 'owned_stop': 16}], 16)
    prediction = np.zeros((16, 3))
    prediction[8:, 0] = 2
    for order, expected in ((2, 2), (3, 3)):
        masks = stencil_masks(np.ones(16, dtype=bool), owners, order)
        derivative = np.diff(prediction, n=order, axis=0)
        assert masks['seam_free'].sum() == expected
        assert np.all(derivative[masks['inside_free']] == 0)
        assert np.all(np.linalg.norm(derivative[masks['seam_free']], axis=-1) > 0)
    free: Array = np.ones(16, dtype=bool)
    free[8] = False
    assert not stencil_masks(free, owners, 2)['seam_free'].any()


def test_constant_bias_has_no_roughness_but_alternating_error_does() -> None:
    error = np.full((32, 3), 4.)
    free: Array = np.ones(32, dtype=bool)
    owners: Array = np.zeros(32, dtype=int)
    result = local_error_components(error, free, owners, 1 / 60)
    np.testing.assert_allclose(result['residual_position'], 0)
    np.testing.assert_allclose(result['error_acceleration'], 0)
    error[:, 0] += .1 * (-1.)**np.arange(32)
    result = local_error_components(error, free, owners, 1 / 60)
    np.testing.assert_allclose(np.linalg.norm(result['error_acceleration'], axis=-1), 1440)
    np.testing.assert_allclose(result['error_acceleration'], result['trend_acceleration'] + result['residual_acceleration'], atol=1e-9)
    assert np.max(np.abs(result['residual_position'])) == pytest.approx(.08)


def test_local_decomposition_never_smooths_across_seams_or_events() -> None:
    error = np.zeros((40, 3))
    error[20:, 0] = 100
    owners: Array = np.repeat([0, 1], 20)
    free: Array = np.ones(40, dtype=bool)
    free[9:12] = False
    error[9:12, 1] = 200
    result = local_error_components(error, free, owners, .1)
    np.testing.assert_allclose(result['error_acceleration'], 0)
    np.testing.assert_allclose(result['residual_acceleration'], 0)
    assert len(result['error_acceleration']) == 19


@pytest.mark.parametrize('windows', [[], [{'owned_start': 1, 'owned_stop': 4}],
                                   [{'owned_start': 0, 'owned_stop': 2}],
                                   [{'owned_start': 0, 'owned_stop': 3}, {'owned_start': 2, 'owned_stop': 4}]])
def test_invalid_ownership_is_rejected(windows: list[dict[str, int]]) -> None:
    with pytest.raises(ValueError, match='ownership'):
        window_owners(windows, 4)
