"""Pure contracts for the query-aware presence residual branch."""

from __future__ import annotations

import pytest
import torch

from src.tasks.plcs.models.components.presence_competition import (
    DeepSetsPresenceResidual,
    build_presence_competition,
)


def test_deepsets_branch_is_exactly_zero_initialized() -> None:
    torch.manual_seed(3)
    branch = DeepSetsPresenceResidual(hidden_dim=8)
    query_hidden = torch.randn(2, 3, 4, 8)

    residual = branch(query_hidden)

    assert residual.shape == (2, 3, 4)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.count_nonzero(branch.output_projection.weight) == 0
    assert torch.count_nonzero(branch.output_projection.bias) == 0


def test_deepsets_branch_is_query_permutation_equivariant_after_training() -> None:
    torch.manual_seed(5)
    branch = DeepSetsPresenceResidual(hidden_dim=6)
    with torch.no_grad():
        branch.output_projection.weight.normal_()
        branch.output_projection.bias.normal_()
    query_hidden = torch.randn(2, 3, 5, 6)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    original = branch(query_hidden)
    permuted = branch(query_hidden[:, :, permutation])

    torch.testing.assert_close(permuted, original[:, :, permutation])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_query_shape_dtype_and_gradients(dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    branch = DeepSetsPresenceResidual(hidden_dim=4).to(dtype=dtype)
    with torch.no_grad():
        branch.output_projection.weight.fill_(0.25)
    query_hidden = torch.randn(2, 3, 1, 4, dtype=dtype, requires_grad=True)

    residual = branch(query_hidden)
    residual.square().mean().backward()

    assert residual.shape == (2, 3, 1)
    assert residual.dtype is dtype
    assert query_hidden.grad is not None
    assert torch.isfinite(query_hidden.grad).all()
    assert bool(query_hidden.grad.abs().any())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in branch.parameters()
    )


def test_zero_initialized_projection_receives_learning_signal() -> None:
    branch = DeepSetsPresenceResidual(hidden_dim=4)
    query_hidden = torch.randn(2, 3, 4, 4)

    branch(query_hidden).sum().backward()

    assert branch.output_projection.weight.grad is not None
    assert bool(branch.output_projection.weight.grad.abs().any())
    assert branch.output_projection.bias.grad is not None
    assert bool(branch.output_projection.bias.grad.abs().any())


def test_feature_projection_receives_gradient_after_zero_output_warmup_step() -> None:
    torch.manual_seed(11)
    branch = DeepSetsPresenceResidual(hidden_dim=4)
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    query_hidden = torch.randn(2, 3, 4, 4)
    target = torch.ones(2, 3, 4)

    optimizer.zero_grad(set_to_none=True)
    first_loss = (branch(query_hidden) - target).square().mean()
    first_loss.backward()
    first_feature_weight_gradient = branch.feature_projection.weight.grad
    assert first_feature_weight_gradient is not None
    assert torch.count_nonzero(first_feature_weight_gradient) == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second_loss = (branch(query_hidden) - target).square().mean()
    second_loss.backward()

    for parameter in (
        branch.feature_projection.weight,
        branch.feature_projection.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert bool(parameter.grad.abs().any())


def test_builder_registers_only_explicit_deepsets_mode() -> None:
    assert build_presence_competition("none", hidden_dim=4) is None
    assert isinstance(
        build_presence_competition("deepsets", hidden_dim=4),
        DeepSetsPresenceResidual,
    )


@pytest.mark.parametrize(
    "query_hidden",
    [torch.randn(2, 3, 4), torch.randn(2, 3, 4, 5)],
)
def test_branch_rejects_invalid_input_contract(query_hidden: torch.Tensor) -> None:
    branch = DeepSetsPresenceResidual(hidden_dim=4)

    with pytest.raises(ValueError, match="query_hidden"):
        branch(query_hidden)
