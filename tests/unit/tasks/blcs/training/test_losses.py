"""Unit tests for BLCS training losses."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.training.losses import BLCSLoss, trajectory_position_loss
from src.utils.schema.court import COURT_COORD_SCALE_Z


def _loss(
    *,
    position_weight: float = 1.0,
    reprojection_weight: float = 0.0,
    position_axis_weights: tuple[float, ...] | None = None,
    smoothness_weight: float = 0.0,
    gravity_weight: float = 0.0,
    smoothness_order: int = 3,
    smoothness_beta: float = 1e-3,
    smoothness_axis_weights: tuple[float, ...] | None = None,
    gravity_beta: float = 5e-3,
    gravity: float = 9.81,
    frame_dt: float = 1.0 / 30.0,
) -> BLCSLoss:
    """Build the complete loss contract with explicit test-owned values."""
    return BLCSLoss(
        position_weight=position_weight,
        reprojection_weight=reprojection_weight,
        position_axis_weights=position_axis_weights,
        smoothness_weight=smoothness_weight,
        gravity_weight=gravity_weight,
        smoothness_order=smoothness_order,
        smoothness_beta=smoothness_beta,
        smoothness_axis_weights=smoothness_axis_weights,
        gravity_beta=gravity_beta,
        gravity=gravity,
        frame_dt=frame_dt,
        height_scale=COURT_COORD_SCALE_Z,
    )


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
        _loss(position_axis_weights=(1.0, 2.0))

    with pytest.raises(ValueError, match="non-negative"):
        _loss(position_axis_weights=(1.0, -1.0, 1.0))


def test_blcs_loss_uses_axis_weights_for_position_term() -> None:
    pred = torch.zeros(1, 1, 3)
    target = torch.tensor([[[0.0, 0.5, 0.0]]])
    loss_fn = _loss(position_axis_weights=(1.0, 4.0, 1.0))

    losses = loss_fn(pred_position=pred, target_position=target)

    unweighted_y = torch.nn.functional.smooth_l1_loss(
        pred[:, :, 1],
        target[:, :, 1],
        reduction="mean",
    )
    expected = unweighted_y * 4.0 / 3.0
    assert losses["position"] == pytest.approx(expected)
    assert losses["total"] == pytest.approx(expected)


def test_zero_weight_physics_priors_are_off() -> None:
    loss_fn = _loss(
        position_weight=1.0,
        smoothness_weight=0.0,
        gravity_weight=0.0,
    )
    pred = torch.randn(1, 20, 3)
    target = torch.randn(1, 20, 3)
    losses = loss_fn(pred_position=pred, target_position=target)
    assert losses["smoothness"].item() == 0.0
    assert losses["gravity"].item() == 0.0


def test_smoothness_prior_penalizes_jitter() -> None:
    loss_fn = _loss(position_weight=0.0, smoothness_weight=1.0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    smooth = loss_fn(pred_position=t, target_position=t)["smoothness"]
    jittery = t + 0.05 * torch.randn(1, 40, 3)
    noisy = loss_fn(pred_position=jittery, target_position=jittery)["smoothness"]
    assert noisy > 10 * smooth


def test_gravity_prior_prefers_ballistic_curvature() -> None:
    loss_fn = _loss(position_weight=0.0, gravity_weight=1.0)
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
    loss_fn = _loss(
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


def test_blcs_loss_rejects_invalid_smoothness_axis_weights() -> None:
    with pytest.raises(ValueError, match="exactly 3 values"):
        _loss(smoothness_weight=1.0, smoothness_axis_weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        _loss(
            smoothness_weight=1.0,
            smoothness_axis_weights=(1.0, 1.0, -1.0),
        )


def test_smoothness_axis_weights_exclude_height_axis() -> None:
    # Trajectory that is smooth in x/y but jittery only on the height (z) axis.
    torch.manual_seed(0)
    traj = torch.zeros(1, 30, 3)
    traj[..., 2] = torch.randn(1, 30)  # z-only jitter
    uniform = _loss(position_weight=0.0, smoothness_weight=1.0)
    # [1, 1, 0] drops the height axis -> the z jitter must not be penalized.
    zeroed_z = _loss(
        position_weight=0.0,
        smoothness_weight=1.0,
        smoothness_axis_weights=(1.0, 1.0, 0.0),
    )
    assert uniform(pred_position=traj, target_position=traj)["smoothness"] > 1e-3
    assert (
        zeroed_z(pred_position=traj, target_position=traj)["smoothness"].item()
        == pytest.approx(0.0, abs=1e-8)
    )


def test_gravity_target_scales_with_output_fps() -> None:
    # dt = 1/fps enters the ballistic target quadratically: halving fps (doubling
    # dt) must quadruple the magnitude of the (negative) target 2nd difference.
    fast = _loss(gravity_weight=1.0, frame_dt=1.0 / 60.0)._gravity_target
    slow = _loss(gravity_weight=1.0, frame_dt=1.0 / 30.0)._gravity_target
    assert fast < 0 and slow < 0
    assert slow == pytest.approx(4.0 * fast, rel=1e-6)
