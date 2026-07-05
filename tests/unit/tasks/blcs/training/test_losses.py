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


def test_physics_priors_off_by_default() -> None:
    loss_fn = BLCSLoss(position_weight=1.0)
    pred = torch.randn(1, 20, 3)
    target = torch.randn(1, 20, 3)
    losses = loss_fn(pred_position=pred, target_position=target)
    assert losses["smoothness"].item() == 0.0
    assert losses["gravity"].item() == 0.0


def test_smoothness_prior_penalizes_jitter() -> None:
    loss_fn = BLCSLoss(position_weight=0.0, smoothness_weight=1.0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    smooth = loss_fn(pred_position=t, target_position=t)["smoothness"]
    jittery = t + 0.05 * torch.randn(1, 40, 3)
    noisy = loss_fn(pred_position=jittery, target_position=jittery)["smoothness"]
    assert noisy > 10 * smooth


def test_gravity_prior_prefers_ballistic_curvature() -> None:
    loss_fn = BLCSLoss(position_weight=0.0, gravity_weight=1.0)
    target_2nd = loss_fn._gravity_target  # normalized ballistic 2nd difference
    steps = torch.arange(40.0)
    xy = torch.zeros(1, 40, 2)
    z_ballistic = (1.0 + 0.02 * steps + 0.5 * target_2nd * steps**2).view(1, 40, 1)
    ballistic = torch.cat([xy, z_ballistic], dim=-1)
    z_flat = (1.0 + 0.02 * steps).view(1, 40, 1)  # zero curvature
    flat = torch.cat([xy, z_flat], dim=-1)
    good = loss_fn(pred_position=ballistic, target_position=ballistic)["gravity"]
    bad = loss_fn(pred_position=flat, target_position=flat)["gravity"]
    assert good < 1e-6
    assert bad > good


def test_physics_priors_contribute_to_total() -> None:
    loss_fn = BLCSLoss(
        position_weight=1.0, smoothness_weight=1.0, gravity_weight=0.5
    )
    pred = torch.randn(1, 30, 3)
    target = torch.randn(1, 30, 3)
    losses = loss_fn(pred_position=pred, target_position=target)
    expected = (
        losses["position"]
        + losses["smoothness"]
        + 0.5 * losses["gravity"]
    )
    assert losses["total"] == pytest.approx(expected.item(), rel=1e-5)
