"""Unit tests for the Court Alignment objectives."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_alignment.training.losses import (
    center_vote_loss,
    centernet_focal_loss,
)


def test_perfect_predictions_have_lower_losses_than_bad_predictions() -> None:
    target = torch.zeros(1, 14, 12, 12)
    target[:, :, 5, 6] = 1.0
    perfect = torch.full_like(target, -15.0)
    perfect[:, :, 5, 6] = 15.0
    bad = -perfect

    assert centernet_focal_loss(perfect, target) < centernet_focal_loss(bad, target)

    vote_target = torch.zeros(1, 2, 12, 12)
    vote_target[:, :, 5, 6] = torch.tensor([2.0, -3.0])
    mask = torch.zeros(1, 12, 12, dtype=torch.bool)
    mask[:, 5, 6] = True
    assert center_vote_loss(vote_target, vote_target, mask) == 0.0
    assert center_vote_loss(torch.zeros_like(vote_target), vote_target, mask) > 0.0


def test_center_vote_mask_can_be_empty_without_nan() -> None:
    prediction = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)

    loss = center_vote_loss(prediction, target, mask)
    assert loss == 0.0
    loss.backward()
    assert prediction.grad is not None


def test_focal_loss_requires_a_lattice_positive_unless_explicitly_allowed() -> None:
    logits = torch.zeros(1, 14, 8, 8)
    target = torch.zeros_like(logits)
    with pytest.raises(ValueError, match="no exact positive"):
        centernet_focal_loss(logits, target)
    assert torch.isfinite(
        centernet_focal_loss(logits, target, allow_no_positive=True)
    )


@pytest.mark.parametrize("bad_mask", [torch.zeros(1, 2, 8, 8), torch.zeros(1, 8)])
def test_center_vote_mask_shape_and_dtype_are_strict(bad_mask: torch.Tensor) -> None:
    prediction = torch.zeros(1, 2, 8, 8)
    with pytest.raises((ValueError, TypeError)):
        center_vote_loss(prediction, prediction, bad_mask)
