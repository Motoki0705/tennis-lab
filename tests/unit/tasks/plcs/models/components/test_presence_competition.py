"""Pure contracts for the query-aware presence residual branch."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

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


def test_default_deepsets_preserves_pre_centering_equation_and_roundtrip() -> None:
    torch.manual_seed(4)
    branch = DeepSetsPresenceResidual(hidden_dim=6)
    restored = DeepSetsPresenceResidual(hidden_dim=6, center_queries=False)
    with torch.no_grad():
        branch.output_projection.weight.normal_()
        branch.output_projection.bias.normal_()
    restored.load_state_dict(branch.state_dict(), strict=True)
    query_hidden = torch.randn(2, 3, 4, 6)
    pooled = query_hidden.mean(dim=-2, keepdim=True).expand_as(query_hidden)
    features = torch.cat(
        (query_hidden, pooled, query_hidden - pooled),
        dim=-1,
    )
    expected = branch.output_projection(
        F.gelu(branch.feature_projection(features))
    ).squeeze(-1)

    assert not branch.center_queries
    assert not restored.center_queries
    assert torch.equal(branch(query_hidden), expected)
    assert torch.equal(restored(query_hidden), expected)
    assert set(branch.state_dict()) == {
        "feature_projection.weight",
        "feature_projection.bias",
        "output_projection.weight",
        "output_projection.bias",
    }


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


def test_centered_branch_is_query_permutation_equivariant_after_training() -> None:
    torch.manual_seed(6)
    branch = DeepSetsPresenceResidual(hidden_dim=6, center_queries=True)
    with torch.no_grad():
        branch.output_projection.weight.normal_()
    query_hidden = torch.randn(2, 3, 5, 6)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    original = branch(query_hidden)
    permuted = branch(query_hidden[:, :, permutation])

    torch.testing.assert_close(permuted, original[:, :, permutation])


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 1.0e-6), (torch.bfloat16, 2.0e-2)],
)
def test_centered_branch_has_zero_query_mean_with_dtype_tolerance(
    dtype: torch.dtype,
    atol: float,
) -> None:
    torch.manual_seed(8)
    branch = DeepSetsPresenceResidual(
        hidden_dim=8,
        center_queries=True,
    ).to(dtype=dtype)
    with torch.no_grad():
        branch.output_projection.weight.normal_()
    query_hidden = torch.randn(2, 3, 4, 8, dtype=dtype)

    residual = branch(query_hidden)

    assert residual.dtype is dtype
    assert torch.isfinite(residual).all()
    torch.testing.assert_close(
        residual.float().mean(dim=-1),
        torch.zeros(2, 3),
        rtol=0.0,
        atol=atol,
    )


def test_centered_single_query_is_strictly_zero_after_training() -> None:
    branch = DeepSetsPresenceResidual(hidden_dim=4, center_queries=True)
    with torch.no_grad():
        branch.output_projection.weight.normal_()
    query_hidden = torch.randn(2, 3, 1, 4)

    residual = branch(query_hidden)

    assert torch.equal(residual, torch.zeros_like(residual))


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


def test_centered_mixed_target_trains_output_then_feature_projection() -> None:
    torch.manual_seed(12)
    branch = DeepSetsPresenceResidual(hidden_dim=4, center_queries=True)
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    query_hidden = torch.randn(2, 3, 4, 4)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 1, 4).expand(2, 3, -1)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in branch.named_parameters()
    }

    optimizer.zero_grad(set_to_none=True)
    first_loss = F.binary_cross_entropy_with_logits(branch(query_hidden), target)
    first_loss.backward()

    output_gradient = branch.output_projection.weight.grad
    assert output_gradient is not None
    assert torch.isfinite(output_gradient).all()
    assert bool(output_gradient.abs().any())
    feature_gradient = branch.feature_projection.weight.grad
    assert feature_gradient is not None
    assert torch.count_nonzero(feature_gradient) == 0
    optimizer.step()

    parameters_after_first_step = dict(branch.named_parameters())
    assert not torch.equal(
        parameters_after_first_step["output_projection.weight"],
        parameters_before["output_projection.weight"],
    )
    for name in (
        "feature_projection.weight",
        "feature_projection.bias",
    ):
        assert torch.equal(parameters_after_first_step[name], parameters_before[name])

    optimizer.zero_grad(set_to_none=True)
    second_loss = F.binary_cross_entropy_with_logits(branch(query_hidden), target)
    second_loss.backward()

    feature_gradient = branch.feature_projection.weight.grad
    assert feature_gradient is not None
    assert torch.isfinite(feature_gradient).all()
    assert bool(feature_gradient.abs().any())


@pytest.mark.parametrize("target_value", [0.0, 1.0])
def test_centered_uniform_presence_target_cancels_common_gradient(
    target_value: float,
) -> None:
    torch.manual_seed(14)
    branch = DeepSetsPresenceResidual(hidden_dim=4, center_queries=True)
    query_hidden = torch.randn(2, 3, 4, 4)
    target = torch.full((2, 3, 4), target_value)

    loss = F.binary_cross_entropy_with_logits(branch(query_hidden), target)
    loss.backward()

    output_gradient = branch.output_projection.weight.grad
    assert output_gradient is not None
    assert torch.isfinite(output_gradient).all()
    torch.testing.assert_close(
        output_gradient,
        torch.zeros_like(output_gradient),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_centered_projection_structurally_omits_common_mode_bias() -> None:
    centered = DeepSetsPresenceResidual(hidden_dim=4, center_queries=True)
    deepsets = DeepSetsPresenceResidual(hidden_dim=4)

    assert centered.output_projection.bias is None
    assert "output_projection.bias" not in dict(centered.named_parameters())
    assert "output_projection.bias" not in centered.state_dict()
    assert set(centered.state_dict()) == {
        "feature_projection.weight",
        "feature_projection.bias",
        "output_projection.weight",
    }
    assert deepsets.output_projection.bias is not None
    assert "output_projection.bias" in dict(deepsets.named_parameters())
    assert "output_projection.bias" in deepsets.state_dict()


def test_builder_registers_only_explicit_deepsets_mode() -> None:
    assert build_presence_competition("none", hidden_dim=4) is None
    assert isinstance(
        build_presence_competition("deepsets", hidden_dim=4),
        DeepSetsPresenceResidual,
    )
    centered = build_presence_competition("deepsets_centered", hidden_dim=4)
    assert isinstance(centered, DeepSetsPresenceResidual)
    assert centered.center_queries
    assert centered.output_projection.bias is None


@pytest.mark.parametrize(
    "query_hidden",
    [torch.randn(2, 3, 4), torch.randn(2, 3, 4, 5)],
)
def test_branch_rejects_invalid_input_contract(query_hidden: torch.Tensor) -> None:
    branch = DeepSetsPresenceResidual(hidden_dim=4)

    with pytest.raises(ValueError, match="query_hidden"):
        branch(query_hidden)
