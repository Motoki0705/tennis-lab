"""Unit tests for standard PLCS loss terms."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from src.tasks.plcs.model_io.contracts import PLCSReprojectionTarget
from src.tasks.plcs.training.losses import (
    DEFAULT_LOSS_TERMS,
    PLCSLoss,
    PLCSLossConfig,
    PLCSLossInputs,
    position_loss_term,
    position_smoothness_loss_term,
    reprojection_loss_term,
)
from src.utils.configuration import SemanticConfigurationError
from src.utils.geometry.court_pose import canonical_pose_to_world_pose
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)
from src.utils.schema.court_normalization import normalize_court_position


def _loss_config() -> PLCSLossConfig:
    return PLCSLossConfig(
        position_weight=1.0,
        position_smooth_l1_beta=1.0,
        rotation_weight=1.0,
        angle_weight=0.0,
        position_smoothness_weight=0.0,
        canonical_pose_weight=0.0,
        reprojection_weight=0.0,
        reprojection_smooth_l1_beta=0.01,
        joint_angle_weight=0.0,
        torsion_angle_weight=0.0,
        torso_twist_weight=0.0,
        bone_length_weight=0.0,
        joint_angle_velocity_weight=0.0,
        torsion_angle_velocity_weight=0.0,
        torso_twist_velocity_weight=0.0,
        joint_angle_velocity_angle_weights=None,
        torsion_angle_velocity_angle_weights=None,
    )


def _reprojection_target(world_pose: torch.Tensor) -> PLCSReprojectionTarget:
    batch_size, frames = world_pose.shape[:2]
    views = 2
    camera_R = torch.eye(3).view(1, 1, 3, 3).expand(
        batch_size, views, -1, -1
    )
    camera_C = torch.tensor(
        [[[0.0, 0.0, -20.0], [1.0, -1.0, -18.0]]],
        dtype=world_pose.dtype,
    ).expand(batch_size, -1, -1)
    camera_f = world_pose.new_tensor([[800.0, 900.0]]).expand(batch_size, -1)
    camera_cx = world_pose.new_full((batch_size, views), 640.0)
    camera_cy = world_pose.new_full((batch_size, views), 360.0)
    camera_w = world_pose.new_full((batch_size, views), 1280.0)
    camera_h = world_pose.new_full((batch_size, views), 720.0)
    target_uv, in_front = DifferentiablePinholeProjection()(
        world_pose,
        camera_R,
        camera_C,
        camera_f,
        camera_cx,
        camera_cy,
        camera_w,
        camera_h,
    )
    assert in_front.all()
    return PLCSReprojectionTarget(
        target_uv=target_uv.detach(),
        target_vis=torch.ones(
            batch_size,
            views,
            frames,
            17,
            dtype=torch.bool,
        ),
        padding_mask=torch.zeros(
            batch_size,
            views,
            frames,
            dtype=torch.bool,
        ),
        camera_R=camera_R,
        camera_C=camera_C,
        camera_f=camera_f,
        camera_cx=camera_cx,
        camera_cy=camera_cy,
        camera_w=camera_w,
        camera_h=camera_h,
    )


def _predicted_pose() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position = torch.tensor(
        [[[0.05, -0.1, 0.02], [0.08, -0.06, 0.03]]],
        requires_grad=True,
    )
    rotation = torch.tensor(
        [[[0.98, 0.2], [0.9, -0.4]]],
        requires_grad=True,
    )
    canonical = (
        torch.linspace(-0.7, 0.9, 2 * 17 * 3)
        .reshape(1, 2, 17, 3)
        .requires_grad_()
    )
    return position, rotation, canonical


def _inputs(pred_position: torch.Tensor, mask: torch.Tensor | None) -> PLCSLossInputs:
    b, *rest = pred_position.shape
    rot_shape = (*pred_position.shape[:-1], 2)
    return PLCSLossInputs(
        pred_position=pred_position,
        pred_rotation=torch.zeros(rot_shape),
        target_position=pred_position,
        target_rotation=torch.zeros(rot_shape),
        frame_mask=mask,
    )


def _position_smoothness(inputs: PLCSLossInputs) -> torch.Tensor:
    return position_smoothness_loss_term(
        inputs,
        penalty=TemporalSmoothnessPenalty(
            order=3,
            beta=1e-3,
            axis_weights=(1.0, 1.0, 1.0),
        ),
    )


def test_registered_term_is_off_when_composed_weight_is_zero() -> None:
    assert "position_smoothness" in DEFAULT_LOSS_TERMS
    config = _loss_config()
    assert config.position_smoothness_weight == 0.0
    assert PLCSLoss(config).weight_for("position_smoothness") == 0.0


def test_from_dict_parses_weight() -> None:
    raw = asdict(_loss_config())
    raw["position_smoothness_weight"] = 4.0
    cfg = PLCSLossConfig.from_dict(raw)
    assert cfg.position_smoothness_weight == 4.0


def test_position_smooth_l1_beta_is_configurable() -> None:
    inputs = PLCSLossInputs(
        pred_position=torch.full((1, 1, 3), 0.2),
        pred_rotation=torch.zeros(1, 1, 2),
        target_position=torch.zeros(1, 1, 3),
        target_rotation=torch.zeros(1, 1, 2),
    )

    torch.testing.assert_close(
        position_loss_term(inputs, beta=1.0), torch.tensor(0.02)
    )
    torch.testing.assert_close(
        position_loss_term(inputs, beta=0.1), torch.tensor(0.15)
    )


def test_position_beta_is_bound_to_combined_loss() -> None:
    loss_fn = PLCSLoss(replace(_loss_config(), position_smooth_l1_beta=0.1))
    inputs = PLCSLossInputs(
        pred_position=torch.full((1, 1, 3), 0.2),
        pred_rotation=torch.zeros(1, 1, 2),
        target_position=torch.zeros(1, 1, 3),
        target_rotation=torch.zeros(1, 1, 2),
    )

    torch.testing.assert_close(
        loss_fn.loss_terms["position"](inputs), torch.tensor(0.15)
    )


def test_position_beta_must_be_positive() -> None:
    raw = asdict(_loss_config())
    raw["position_smooth_l1_beta"] = 0.0
    with pytest.raises(SemanticConfigurationError, match="must be positive"):
        PLCSLossConfig.from_dict(raw)


def test_reprojection_beta_must_be_positive() -> None:
    raw = asdict(_loss_config())
    raw["reprojection_smooth_l1_beta"] = 0.0
    with pytest.raises(SemanticConfigurationError, match="must be positive"):
        PLCSLossConfig.from_dict(raw)


def test_reprojection_is_zero_for_the_pose_that_generated_the_target() -> None:
    position, rotation, canonical = _predicted_pose()
    world_pose = canonical_pose_to_world_pose(canonical, position, rotation)
    inputs = PLCSLossInputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position.detach(),
        target_rotation=rotation.detach(),
        pred_canonical_pose=canonical,
        reprojection_target=_reprojection_target(world_pose),
    )

    loss = reprojection_loss_term(
        inputs,
        projector=DifferentiablePinholeProjection(),
        beta=0.01,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_reprojection_backpropagates_to_position_rotation_and_canonical_pose() -> None:
    position, rotation, canonical = _predicted_pose()
    world_pose = canonical_pose_to_world_pose(canonical, position, rotation)
    target = _reprojection_target(world_pose)
    offset = torch.linspace(
        -0.01,
        0.01,
        target.target_uv.numel(),
    ).reshape_as(target.target_uv)
    target = replace(target, target_uv=target.target_uv + offset)
    inputs = PLCSLossInputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position.detach(),
        target_rotation=rotation.detach(),
        pred_canonical_pose=canonical,
        reprojection_target=target,
    )

    reprojection_loss_term(
        inputs,
        projector=DifferentiablePinholeProjection(),
        beta=0.01,
    ).backward()

    for prediction in (position, rotation, canonical):
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
        assert prediction.grad.abs().sum() > 0


def test_reprojection_ignores_invisible_joints_and_padded_views() -> None:
    position, rotation, canonical = _predicted_pose()
    world_pose = canonical_pose_to_world_pose(canonical, position, rotation)
    target = _reprojection_target(world_pose)
    target_uv = target.target_uv.clone()
    target_vis = target.target_vis.clone()
    padding_mask = target.padding_mask.clone()
    target_uv[:, 0, 0, 0] += 0.5
    target_vis[:, 0, 0, 0] = False
    target_uv[:, 1, 1] += 0.5
    padding_mask[:, 1, 1] = True
    target = replace(
        target,
        target_uv=target_uv,
        target_vis=target_vis,
        padding_mask=padding_mask,
    )
    inputs = PLCSLossInputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position.detach(),
        target_rotation=rotation.detach(),
        pred_canonical_pose=canonical,
        reprojection_target=target,
    )

    loss = reprojection_loss_term(
        inputs,
        projector=DifferentiablePinholeProjection(),
        beta=0.01,
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_reprojection_does_not_mask_a_prediction_behind_the_camera() -> None:
    position, rotation, canonical = _predicted_pose()
    world_pose = canonical_pose_to_world_pose(canonical, position, rotation)
    target = _reprojection_target(world_pose)
    target = replace(
        target,
        target_uv=torch.zeros_like(target.target_uv),
        camera_C=torch.tensor(
            [[[0.0, 0.0, 20.0], [1.0, -1.0, 18.0]]]
        ),
    )
    inputs = PLCSLossInputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position.detach(),
        target_rotation=rotation.detach(),
        pred_canonical_pose=canonical,
        reprojection_target=target,
    )

    loss = reprojection_loss_term(
        inputs,
        projector=DifferentiablePinholeProjection(),
        beta=0.01,
    )

    assert loss > 0


def test_reprojection_weight_contributes_to_combined_loss() -> None:
    position, rotation, canonical = _predicted_pose()
    world_pose = canonical_pose_to_world_pose(canonical, position, rotation)
    target = _reprojection_target(world_pose)
    target = replace(target, target_uv=target.target_uv + 0.01)
    config = replace(
        _loss_config(),
        position_weight=0.0,
        rotation_weight=0.0,
        reprojection_weight=2.0,
    )
    loss_fn = PLCSLoss(config)
    prepared = loss_fn.prepare_inputs(
        pred_position=position,
        pred_rotation=rotation,
        target_position=position.detach(),
        target_rotation=rotation.detach(),
        pred_canonical_pose=canonical,
        target_human_kp_3d=None,
        padding_mask=target.padding_mask,
        reprojection_target=target,
    )

    losses = loss_fn(prepared)

    assert losses["reprojection"] > 0
    torch.testing.assert_close(losses["total"], 2.0 * losses["reprojection"])


def test_enabled_reprojection_requires_pose_and_target_bundle() -> None:
    loss_fn = PLCSLoss(replace(_loss_config(), reprojection_weight=1.0))
    position = torch.zeros(1, 1, 3)
    rotation = torch.tensor([[[1.0, 0.0]]])
    with pytest.raises(ValueError, match="pred_canonical_pose"):
        loss_fn.prepare_inputs(
            pred_position=position,
            pred_rotation=rotation,
            target_position=position,
            target_rotation=rotation,
            pred_canonical_pose=None,
            target_human_kp_3d=None,
            padding_mask=None,
        )
    with pytest.raises(ValueError, match="reprojection_target"):
        loss_fn.prepare_inputs(
            pred_position=position,
            pred_rotation=rotation,
            target_position=position,
            target_rotation=rotation,
            pred_canonical_pose=torch.zeros(1, 1, 17, 3),
            target_human_kp_3d=None,
            padding_mask=None,
        )


def test_frame_level_input_is_noop() -> None:
    # (B, 3): no temporal axis -> term must be a no-op, not misread coords as time.
    pred = torch.randn(5, 3)
    assert _position_smoothness(_inputs(pred, None)).item() == 0.0


def test_jitter_penalized_more_than_smooth() -> None:
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    mask = torch.ones(1, 40, dtype=torch.bool)
    smooth = _position_smoothness(_inputs(t, mask))
    jittery = _position_smoothness(_inputs(t + 0.02 * torch.randn(1, 40, 3), mask))
    assert jittery > 10 * smooth


def test_contributes_to_total_via_forward() -> None:
    torch.manual_seed(1)
    cfg = replace(
        _loss_config(),
        position_weight=0.0,
        rotation_weight=0.0,
        position_smoothness_weight=4.0,
    )
    loss_fn = PLCSLoss(config=cfg)
    pred_pos = torch.randn(1, 20, 3)
    pred_rot = torch.randn(1, 20, 2)
    inputs = loss_fn.prepare_inputs(
        pred_position=pred_pos,
        pred_rotation=pred_rot,
        target_position=pred_pos,
        target_rotation=pred_rot,
        pred_canonical_pose=None,
        target_human_kp_3d=None,
        padding_mask=torch.zeros(1, 20, dtype=torch.bool),
    )
    losses = loss_fn(inputs)
    assert losses["position_smoothness"] > 0
    torch.testing.assert_close(losses["total"], 4.0 * losses["position_smoothness"])


def test_forward_combines_only_prepared_terms_without_registry_dispatch() -> None:
    loss_fn = PLCSLoss(config=_loss_config())
    pred_pos = torch.randn(1, 5, 3)
    pred_rot = torch.randn(1, 5, 2)
    prepared = loss_fn.prepare_inputs(
        pred_position=pred_pos,
        pred_rotation=pred_rot,
        target_position=pred_pos,
        target_rotation=pred_rot,
        pred_canonical_pose=None,
        target_human_kp_3d=None,
        padding_mask=torch.zeros(1, 5, dtype=torch.bool),
    )
    loss_fn.loss_terms.clear()

    losses = loss_fn(prepared)

    assert torch.isfinite(losses["total"])


def test_default_position_loss_is_equal_for_same_physical_xyz_error() -> None:
    loss_fn = PLCSLoss(config=_loss_config())
    target_position = torch.zeros(1, 1, 3)
    rotation = torch.tensor([[[1.0, 0.0]]])
    values = []
    for axis in range(3):
        physical_error = torch.zeros(1, 1, 3)
        physical_error[..., axis] = 0.25
        prepared = loss_fn.prepare_inputs(
            pred_position=normalize_court_position(physical_error),
            pred_rotation=rotation,
            target_position=target_position,
            target_rotation=rotation,
            pred_canonical_pose=None,
            target_human_kp_3d=None,
            padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        values.append(loss_fn(prepared)["position"])

    torch.testing.assert_close(torch.stack(values), values[0].expand(3))
