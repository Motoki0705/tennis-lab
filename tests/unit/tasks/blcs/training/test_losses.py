"""Unit tests for BLCS training losses."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.training.losses import BLCSLoss, trajectory_position_loss


def test_trajectory_position_loss_accepts_axis_weights() -> None:
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor(
        [
            [
                [0.2, 0.4, 0.6],
                [0.8, 0.5, 0.3],
            ]
        ]
    )
    mask = torch.tensor([[1.0, 0.0]])
    axis_weights = torch.tensor([1.0, 4.0, 2.0])

    actual = trajectory_position_loss(
        pred,
        target,
        mask,
        axis_weights=axis_weights,
    )

    per_axis = torch.nn.functional.smooth_l1_loss(
        pred[:, :1],
        target[:, :1],
        reduction="none",
    )
    expected = (per_axis * axis_weights.view(1, 1, 3)).mean()
    assert actual == pytest.approx(expected)


def test_blcs_loss_rejects_invalid_axis_weights() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        BLCSLoss(position_axis_weights=(1.0, 2.0))

    with pytest.raises(ValueError, match="non-negative"):
        BLCSLoss(position_axis_weights=(1.0, -1.0, 1.0))


def test_blcs_loss_uses_axis_weights_for_position_term() -> None:
    pred = torch.zeros(1, 1, 3)
    target = torch.tensor([[[0.0, 0.5, 0.0]]])
    loss_fn = BLCSLoss(position_axis_weights=(1.0, 4.0, 1.0))

    losses = loss_fn(pred_position=pred, target_position=target)

    unweighted_y = torch.nn.functional.smooth_l1_loss(
        pred[:, :, 1],
        target[:, :, 1],
        reduction="mean",
    )
    expected = unweighted_y * 4.0 / 3.0
    assert losses["position"] == pytest.approx(expected)
    assert losses["total"] == pytest.approx(expected)
