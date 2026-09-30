"""Read-out identifiability, rank reporting, and physical error aggregation."""
import pytest
import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.conditioning_probe import (
    error_summary,
    fit_readout,
)


def test_linear_readout_recovers_affine_signal_on_unseen_frames() -> None:
    rng = torch.Generator().manual_seed(936)
    train = torch.randn(40, 7, dtype=torch.float64, generator=rng)
    val = torch.randn(13, 7, dtype=torch.float64, generator=rng)
    weight = torch.randn(7, 3, dtype=torch.float64, generator=rng)
    bias = torch.tensor([1., -3., 2.], dtype=torch.float64)
    fitted = fit_readout(train, train @ weight + bias)
    assert fitted.rank == 8
    torch.testing.assert_close(fitted.predict(val), val @ weight + bias, atol=1e-12, rtol=1e-12)


def test_rank_deficiency_is_explicit_and_uses_minimum_norm_solution() -> None:
    x = torch.arange(12, dtype=torch.float64)
    tokens = torch.stack((x, x, torch.zeros_like(x)), dim=-1)
    targets = torch.stack((2*x + 1, x - 3, -x), dim=-1)
    fitted = fit_readout(tokens, targets)
    assert fitted.rank == 2
    assert len(fitted.singular_values) == 4
    torch.testing.assert_close(fitted.predict(tokens), targets)
    torch.testing.assert_close(fitted.coefficients[0], fitted.coefficients[1])


@pytest.mark.parametrize('kind', ['nan_token', 'inf_target', 'shape'])
def test_invalid_fit_is_rejected(kind: str) -> None:
    tokens, targets = torch.zeros(10, 4), torch.zeros(10, 3)
    if kind == 'nan_token':
        tokens[0, 0] = float('nan')
    elif kind == 'inf_target':
        targets[0, 0] = float('inf')
    else:
        targets = targets[:-1]
    with pytest.raises(ValueError):
        fit_readout(tokens, targets)


def test_error_uses_euclidean_metres_and_preserves_empty_strata() -> None:
    target = torch.zeros(2, 3, dtype=torch.float64)
    predicted = torch.tensor([[3., 4., 0.], [0., 0., 0.]], dtype=torch.float64)
    result = error_summary(predicted, target)
    assert result['frames'] == 2
    assert result['rmse_m'] == pytest.approx((25/2)**.5)
    assert result['p95_m'] == pytest.approx(4.75)
    assert error_summary(predicted[:0], target[:0]) == {'frames': 0, 'rmse_m': None, 'p95_m': None, 'maximum_m': None}
