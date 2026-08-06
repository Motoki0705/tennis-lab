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
    loss = FocalBCEWithLogitsLoss(gamma=0.0)
    logits = torch.tensor([[2.0, -1.0], [0.5, -3.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    got = loss(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(got, expected, atol=1e-6)


def test_focal_downweights_easy_examples() -> None:
    """A confident-correct prediction contributes less with gamma>0 than gamma=0."""
    logits = torch.tensor([[6.0]])  # very confident
    targets = torch.tensor([[1.0]])  # correct

    plain = FocalBCEWithLogitsLoss(gamma=0.0)(logits, targets)
    focal = FocalBCEWithLogitsLoss(gamma=2.0)(logits, targets)
    assert focal < plain


def test_matches_manual_formula() -> None:
    """Loss equals mean((1 - p_t)**gamma * BCE)."""
    gamma = 1.5
    loss = FocalBCEWithLogitsLoss(gamma=gamma)
    logits = torch.tensor([0.3, -0.7, 1.2])
    targets = torch.tensor([1.0, 0.0, 1.0])

    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1 - probs) * (1 - targets)
    expected = ((1 - p_t) ** gamma * bce).mean()

    assert torch.allclose(loss(logits, targets), expected, atol=1e-6)


def test_loss_is_non_negative_scalar() -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0)
    out = loss(torch.randn(4, 5), torch.randint(0, 2, (4, 5)).float())
    assert out.ndim == 0
    assert out.item() >= 0.0


def test_negative_gamma_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FocalBCEWithLogitsLoss(gamma=-0.1)


def test_boundary_validator_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same"):
        validate_focal_bce_inputs(torch.randn(2, 3), torch.randn(2, 4))


def test_perfect_prediction_near_zero_loss() -> None:
    loss = FocalBCEWithLogitsLoss(gamma=2.0)
    logits = torch.tensor([[20.0, -20.0]])
    targets = torch.tensor([[1.0, 0.0]])
    assert loss(logits, targets).item() < 1e-6
