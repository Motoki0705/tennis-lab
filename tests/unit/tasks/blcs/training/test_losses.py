"""Unit tests for BLCS training losses."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.tasks.blcs.models.components.differentiable_projection import (
    DifferentiableProjection,
)
from src.tasks.blcs.training.losses import (
    BLCSLoss,
    reprojection_loss,
    trajectory_position_loss,
)
from src.utils.schema.court import COURT_COORD_SCALE_Z
from src.utils.schema.court_normalization import normalize_court_position


class _StaticProjection(DifferentiableProjection):
    """Return fixed projection tensors so reduction semantics stay isolated."""

    def __init__(self, uv: Tensor, in_front: Tensor) -> None:
        super().__init__()
        self.uv = uv
        self.in_front = in_front

    def forward(
        self,
        position_norm: Tensor,
        camera_R: Tensor,
        camera_C: Tensor,
        camera_f: Tensor,
        camera_cx: Tensor,
        camera_cy: Tensor,
        camera_w: Tensor,
        camera_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del (
            position_norm,
            camera_R,
            camera_C,
            camera_f,
            camera_cx,
            camera_cy,
            camera_w,
            camera_h,
        )
        return self.uv, self.in_front


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


def _loss_call(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the complete, already-normalized BLCS loss boundary."""
    batch_size, frames = pred.shape[:2]
    scalar = torch.ones(batch_size, 1, dtype=pred.dtype)
    return {
        "pred_position": pred,
        "target_position": target,
        "mask": torch.ones(batch_size, frames, dtype=torch.bool),
        "target_uv": torch.zeros(batch_size, 1, frames, 2, dtype=pred.dtype),
        "target_vis": torch.zeros(batch_size, 1, frames, dtype=torch.bool),
        "camera_R": torch.eye(3, dtype=pred.dtype)
        .view(1, 1, 3, 3)
        .expand(batch_size, -1, -1, -1),
        "camera_C": torch.tensor([0.0, 0.0, -20.0], dtype=pred.dtype)
        .view(1, 1, 3)
        .expand(batch_size, -1, -1),
        "camera_f": scalar,
        "camera_cx": scalar,
        "camera_cy": scalar,
        "camera_w": scalar,
        "camera_h": scalar,
    }


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


def test_trajectory_position_loss_returns_zero_for_empty_mask() -> None:
    actual = trajectory_position_loss(
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.zeros(1, 2, dtype=torch.bool),
        axis_weights=torch.ones(3),
    )

    assert torch.isfinite(actual)
    assert actual.item() == 0.0


def test_reprojection_loss_reduces_only_explicitly_expanded_mask_entries() -> None:
    pred_uv = torch.zeros(1, 2, 2, 2)
    target_uv = torch.tensor([[[[0.2, 0.4], [0.8, 0.6]], [[0.3, 0.7], [0.9, 0.5]]]])
    projector = _StaticProjection(
        pred_uv,
        torch.ones(1, 2, 2, dtype=torch.bool),
    )
    unused_scalar = torch.ones(1, 2)

    actual = reprojection_loss(
        pred_position=torch.zeros(1, 2, 3),
        target_uv=target_uv,
        target_vis=torch.tensor([[[True, True], [False, True]]]),
        camera_R=torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1),
        camera_C=torch.zeros(1, 2, 3),
        camera_f=unused_scalar,
        camera_cx=unused_scalar,
        camera_cy=unused_scalar,
        camera_w=unused_scalar,
        camera_h=unused_scalar,
        projector=projector,
        mask=torch.tensor([[True, False]]),
    )
    expected = torch.nn.functional.smooth_l1_loss(
        pred_uv[:, :1, :1],
        target_uv[:, :1, :1],
        reduction="mean",
    )

    assert actual == pytest.approx(expected)

    empty = reprojection_loss(
        pred_position=torch.zeros(1, 2, 3),
        target_uv=target_uv,
        target_vis=torch.zeros(1, 2, 2, dtype=torch.bool),
        camera_R=torch.eye(3).view(1, 1, 3, 3).expand(1, 2, -1, -1),
        camera_C=torch.zeros(1, 2, 3),
        camera_f=unused_scalar,
        camera_cx=unused_scalar,
        camera_cy=unused_scalar,
        camera_w=unused_scalar,
        camera_h=unused_scalar,
        projector=projector,
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.isfinite(empty)
    assert empty.item() == 0.0


def test_blcs_loss_rejects_invalid_axis_weights() -> None:
    with pytest.raises(ValueError, match="exactly 3"):
        _loss(position_axis_weights=(1.0, 2.0))

    with pytest.raises(ValueError, match="non-negative"):
        _loss(position_axis_weights=(1.0, -1.0, 1.0))


def test_removed_gravity_target_compatibility_accessor_is_unavailable() -> None:
    loss_fn = _loss(gravity_weight=1.0)

    assert "_gravity_target" not in BLCSLoss.__dict__
    with pytest.raises(AttributeError, match="_gravity_target"):
        _ = loss_fn._gravity_target  # type: ignore[attr-defined]


def test_blcs_loss_uses_axis_weights_for_position_term() -> None:
    pred = torch.zeros(1, 1, 3)
    target = torch.tensor([[[0.0, 0.5, 0.0]]])
    loss_fn = _loss(position_axis_weights=(1.0, 4.0, 1.0))

    losses = loss_fn(**_loss_call(pred, target))

    unweighted_y = torch.nn.functional.smooth_l1_loss(
        pred[:, :, 1],
        target[:, :, 1],
        reduction="mean",
    )
    expected = unweighted_y * 4.0 / 3.0
    assert losses["position"] == pytest.approx(expected)
    assert losses["total"] == pytest.approx(expected)


def test_default_position_loss_is_equal_for_same_physical_xyz_error() -> None:
    loss_fn = _loss(position_axis_weights=None)
    target = torch.zeros(1, 1, 3)
    losses = []
    for axis in range(3):
        physical_error = torch.zeros(1, 1, 3)
        physical_error[..., axis] = 0.25
        pred = normalize_court_position(physical_error)
        losses.append(loss_fn(**_loss_call(pred, target))["position"])

    torch.testing.assert_close(torch.stack(losses), losses[0].expand(3))


def test_zero_weight_physics_priors_are_off() -> None:
    loss_fn = _loss(
        position_weight=1.0,
        smoothness_weight=0.0,
        gravity_weight=0.0,
    )
    pred = torch.randn(1, 20, 3)
    target = torch.randn(1, 20, 3)
    losses = loss_fn(**_loss_call(pred, target))
    assert losses["smoothness"].item() == 0.0
    assert losses["gravity"].item() == 0.0


def test_smoothness_prior_penalizes_jitter() -> None:
    loss_fn = _loss(position_weight=0.0, smoothness_weight=1.0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    smooth = loss_fn(**_loss_call(t, t))["smoothness"]
    jittery = t + 0.05 * torch.randn(1, 40, 3)
    noisy = loss_fn(**_loss_call(jittery, jittery))["smoothness"]
    assert noisy > 10 * smooth


def test_gravity_prior_prefers_ballistic_curvature() -> None:
    loss_fn = _loss(position_weight=0.0, gravity_weight=1.0)
    target_2nd = loss_fn.gravity_penalty.target_second_difference
    steps = torch.arange(40.0)
    xy = torch.zeros(1, 40, 2)
    z_ballistic = (1.0 + 0.02 * steps + 0.5 * target_2nd * steps**2).view(1, 40, 1)
    ballistic = torch.cat([xy, z_ballistic], dim=-1)
    z_flat = (1.0 + 0.02 * steps).view(1, 40, 1)  # zero curvature
    flat = torch.cat([xy, z_flat], dim=-1)
    good = loss_fn(**_loss_call(ballistic, ballistic))["gravity"]
    bad = loss_fn(**_loss_call(flat, flat))["gravity"]
    assert good < 1e-6
    assert bad > good


def test_physics_priors_contribute_to_total() -> None:
    loss_fn = _loss(position_weight=1.0, smoothness_weight=1.0, gravity_weight=0.5)
    pred = torch.randn(1, 30, 3)
    target = torch.randn(1, 30, 3)
    losses = loss_fn(**_loss_call(pred, target))
    expected = losses["position"] + losses["smoothness"] + 0.5 * losses["gravity"]
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
    assert uniform(**_loss_call(traj, traj))["smoothness"] > 1e-3
    assert zeroed_z(**_loss_call(traj, traj))["smoothness"].item() == pytest.approx(
        0.0, abs=1e-8
    )


def test_ballistic_target_second_difference_scales_with_output_fps() -> None:
    # dt = 1/fps enters the ballistic target quadratically: halving fps (doubling
    # dt) must quadruple the magnitude of the (negative) target 2nd difference.
    fast = _loss(
        gravity_weight=1.0,
        frame_dt=1.0 / 60.0,
    ).gravity_penalty.target_second_difference
    slow = _loss(
        gravity_weight=1.0,
        frame_dt=1.0 / 30.0,
    ).gravity_penalty.target_second_difference
    assert fast < 0 and slow < 0
    assert slow == pytest.approx(4.0 * fast, rel=1e-6)
