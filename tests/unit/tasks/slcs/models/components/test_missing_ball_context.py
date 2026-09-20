"""Court retention uses only observed court on real missing-ball frames."""

import pytest
import torch

from src.tasks.slcs.models.components.missing_ball_context import (
    MissingBallCourtContext,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_context_mask_coordinates_flags_and_gradients(dtype: torch.dtype) -> None:
    context = MissingBallCourtContext(num_court_kp=2, dim=3).to(dtype=dtype)
    court = torch.tensor(
        [[[[0.2, 0.4], [0.7, 0.8]]] * 4], dtype=dtype, requires_grad=True
    )
    # Real missing ball / visible ball / all court missing / padding.
    court_vis = torch.tensor([[[1, 0], [1, 1], [0, 0], [1, 1]]])
    ball_vis = torch.tensor([[False, True, False, False]])
    padding = torch.tensor([[False, False, False, True]])
    with torch.no_grad():
        context.weight.fill_(1)
    result = context(court, court_vis, ball_vis, padding)
    torch.testing.assert_close(result[0, 0], torch.full((3,), 1.6, dtype=dtype))
    assert torch.equal(result[0, 1:], torch.zeros(3, 3, dtype=dtype))
    changed = court.detach().clone()
    changed[~(court_vis > 0)] = float("nan")
    assert torch.equal(result, context(changed, court_vis, ball_vis, padding))
    changed[0, 0, 0, 0] += 1
    assert not torch.equal(result[0, 0], context(changed, court_vis, ball_vis, padding)[0, 0])
    result.sum().backward()
    assert context.weight.grad is not None
    assert torch.isfinite(context.weight.grad).all()
    assert context.weight.grad.abs().sum() > 0
    assert court.grad is not None
    assert court.grad[0, 0, 0].abs().sum() > 0
    assert court.grad[0, 0, 1].abs().sum() == 0
    assert court.grad[0, 1:].abs().sum() == 0


def test_projection_initializes_to_zero_without_advancing_rng() -> None:
    state = torch.get_rng_state()
    context = MissingBallCourtContext(num_court_kp=14, dim=32)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.count_nonzero(context.weight) == 0


@pytest.mark.parametrize("court,dim", [(0, 3), (2, 0), (-1, 3)])
def test_invalid_dimensions_rejected(court: int, dim: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MissingBallCourtContext(num_court_kp=court, dim=dim)
