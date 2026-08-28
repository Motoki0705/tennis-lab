"""Unit tests for PLCS loss terms."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from src.tasks.plcs.training.losses import (
    DEFAULT_LOSS_TERMS,
    PLCSLoss,
    PLCSLossConfig,
    PLCSLossInputs,
    canonical_pose_loss_term,
    position_smoothness_loss_term,
)
from src.utils.configuration import SemanticConfigurationError
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.schema.court_normalization import normalize_court_position


def _loss_config() -> PLCSLossConfig:
    return PLCSLossConfig(
        position_weight=1.0,
        rotation_weight=1.0,
        angle_weight=0.0,
        position_smoothness_weight=0.0,
        canonical_pose_weight=0.0,
        canonical_pose_smooth_l1_beta=1.0,
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


def test_canonical_pose_smooth_l1_beta_is_configurable_in_metres() -> None:
    target_pose = torch.zeros(1, 1, 1, 3)
    pred_pose = torch.full_like(target_pose, 0.2)
    inputs = PLCSLossInputs(
        pred_position=torch.zeros(1, 1, 3),
        pred_rotation=torch.zeros(1, 1, 2),
        target_position=torch.zeros(1, 1, 3),
        target_rotation=torch.zeros(1, 1, 2),
        pred_canonical_pose=pred_pose,
        target_canonical_pose=target_pose,
    )

    # smooth-L1(0.2; beta=1.0) = 0.5 * 0.2**2, while the beta=0.1
    # branch is linear: 0.2 - 0.5 * 0.1.
    torch.testing.assert_close(
        canonical_pose_loss_term(inputs, beta=1.0), torch.tensor(0.02)
    )
    torch.testing.assert_close(
        canonical_pose_loss_term(inputs, beta=0.1), torch.tensor(0.15)
    )


def test_canonical_pose_beta_is_bound_to_combined_loss() -> None:
    raw = asdict(_loss_config())
    raw.update(
        canonical_pose_weight=1.0,
        canonical_pose_smooth_l1_beta=0.1,
    )
    loss_fn = PLCSLoss(PLCSLossConfig.from_dict(raw))
    target_pose = torch.zeros(1, 1, 1, 3)
    pred_pose = torch.full_like(target_pose, 0.2)
    inputs = PLCSLossInputs(
        pred_position=torch.zeros(1, 1, 3),
        pred_rotation=torch.zeros(1, 1, 2),
        target_position=torch.zeros(1, 1, 3),
        target_rotation=torch.zeros(1, 1, 2),
        pred_canonical_pose=pred_pose,
        target_canonical_pose=target_pose,
    )

    torch.testing.assert_close(
        loss_fn.loss_terms["canonical_pose"](inputs), torch.tensor(0.15)
    )


def test_canonical_pose_beta_must_be_positive() -> None:
    raw = asdict(_loss_config())
    raw["canonical_pose_smooth_l1_beta"] = 0.0
    with pytest.raises(SemanticConfigurationError, match="must be positive"):
        PLCSLossConfig.from_dict(raw)


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
