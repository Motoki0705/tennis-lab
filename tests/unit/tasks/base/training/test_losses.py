"""Unit tests for the shared focal BCE loss."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.tasks.base.training.losses import (
    FocalBCEWithLogitsLoss,
    validate_focal_bce_inputs,
)

pytestmark = pytest.mark.unit


def test_gamma_zero_equals_mean_bce() -> None:
    """With gamma=0 the focal modulation vanishes, leaving plain mean BCE."""
    loss = FocalBCEWithLogitsLoss(gamma=0.0, positive_weight=1.0)
    logits = torch.tensor([[2.0, -1.0], [0.5, -3.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    got = loss(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(got, expected, atol=1e-6)


def test_focal_downweights_easy_examples() -> None:
    """A confident-correct prediction contributes less with gamma>0 than gamma=0."""
    logits = torch.tensor([[6.0]])  # very confident
    targets = torch.tensor([[1.0]])  # correct

    plain = FocalBCEWithLogitsLoss(gamma=0.0, positive_weight=1.0)(logits, targets)
    focal = FocalBCEWithLogitsLoss(gamma=2.0, positive_weight=1.0)(logits, targets)
    assert focal < plain


def test_matches_manual_formula() -> None:
    """Loss equals mean((1 - p_t)**gamma * BCE)."""
    gamma = 1.5
    loss = FocalBCEWithLogitsLoss(gamma=gamma, positive_weight=1.0)
    logits = torch.tensor([0.3, -0.7, 1.2])
    targets = torch.tensor([1.0, 0.0, 1.0])

    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    expected = ((1 - p_t) ** gamma * bce).mean()

    assert torch.allclose(loss(logits, targets), expected, atol=1e-6)


def test_loss_is_non_negative_scalar() -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0, positive_weight=1.0)
    out = loss(torch.randn(4, 5), torch.randint(0, 2, (4, 5)).float())
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_negative_gamma_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FocalBCEWithLogitsLoss(gamma=-0.1, positive_weight=1.0)


@pytest.mark.parametrize("positive_weight", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_positive_weight_raises(positive_weight: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        FocalBCEWithLogitsLoss(gamma=2.0, positive_weight=positive_weight)


def test_boundary_validator_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same"):
        validate_focal_bce_inputs(torch.randn(2, 3), torch.randn(2, 4))


def test_perfect_prediction_near_zero_loss() -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0, positive_weight=1.0)
    logits = torch.tensor([[20.0, -20.0]])
    targets = torch.tensor([[1.0, 0.0]])
    assert loss(logits, targets).item() < 1e-6


def test_unit_positive_weight_preserves_soft_target_focal_loss() -> None:
    logits = torch.tensor([-1.2, 0.3, 1.7])
    targets = torch.tensor([0.0, 0.35, 1.0])
    gamma = 1.5
    loss = FocalBCEWithLogitsLoss(gamma=gamma, positive_weight=1.0)

    probabilities = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    expected = ((1.0 - p_t) ** gamma * bce).mean()

    torch.testing.assert_close(loss(logits, targets), expected)


def test_positive_weight_amplifies_positive_loss_and_gradient_only() -> None:
    def contribution(target: float, positive_weight: float) -> tuple[float, float]:
        logit = torch.zeros(1, requires_grad=True)
        loss = FocalBCEWithLogitsLoss(
            gamma=0.0,
            positive_weight=positive_weight,
        )(logit, torch.tensor([target]))
        loss.backward()
        assert logit.grad is not None
        return float(loss.item()), float(logit.grad.item())

    positive_base = contribution(1.0, 1.0)
    positive_weighted = contribution(1.0, 4.0)
    negative_base = contribution(0.0, 1.0)
    negative_weighted = contribution(0.0, 4.0)

    assert positive_weighted[0] == pytest.approx(4.0 * positive_base[0])
    assert positive_weighted[1] == pytest.approx(4.0 * positive_base[1])
    assert negative_weighted == pytest.approx(negative_base)


def test_positive_weight_uses_standard_bce_semantics_for_soft_targets() -> None:
    logits = torch.tensor([-0.8, 0.4, 1.1])
    targets = torch.tensor([0.2, 0.55, 0.9])
    positive_weight = 3.0
    gamma = 2.0
    loss = FocalBCEWithLogitsLoss(
        gamma=gamma,
        positive_weight=positive_weight,
    )

    probabilities = torch.sigmoid(logits)
    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor(positive_weight),
        reduction="none",
    )
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    expected = ((1.0 - p_t) ** gamma * weighted_bce).mean()

    torch.testing.assert_close(loss(logits, targets), expected)
