"""Unit tests for the PLCS position-smoothness loss term."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from src.tasks.plcs.training.losses import (
    DEFAULT_LOSS_TERMS,
    PLCSLoss,
    PLCSLossConfig,
    PLCSLossInputs,
    position_smoothness_loss_term,
)
from src.tasks.plcs.training.tracking_losses import PLCSTrackingLoss
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization


def _loss_config() -> PLCSLossConfig:
    return PLCSLossConfig(
        position_weight=1.0,
        rotation_weight=1.0,
        angle_weight=0.0,
        position_smoothness_weight=0.0,
        canonical_pose_weight=0.0,
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


def test_frame_level_input_is_noop() -> None:
    # (B, 3): no temporal axis -> term must be a no-op, not misread coords as time.
    pred = torch.randn(5, 3)
    assert _position_smoothness(_inputs(pred, None)).item() == 0.0


def test_jitter_penalized_more_than_smooth() -> None:
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    mask = torch.ones(1, 40, dtype=torch.bool)
    smooth = _position_smoothness(_inputs(t, mask))
    jittery = _position_smoothness(
        _inputs(t + 0.02 * torch.randn(1, 40, 3), mask)
    )
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


def test_v2_equal_physical_axis_errors_have_equal_default_position_loss() -> None:
    contract = resolve_court_coordinate_normalization("v2")
    loss_fn = PLCSLoss(_loss_config(), normalization=contract)
    losses = []
    for axis in range(3):
        target = torch.zeros(1, 1, 3)
        target[..., axis] = 0.5 / contract.scale_xyz[axis]
        prepared = loss_fn.prepare_inputs(
            pred_position=torch.zeros_like(target),
            pred_rotation=torch.tensor([[[1.0, 0.0]]]),
            target_position=target,
            target_rotation=torch.tensor([[[1.0, 0.0]]]),
            pred_canonical_pose=None,
            target_human_kp_3d=None,
            padding_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        losses.append(loss_fn(prepared)["position"])

    assert loss_fn.position_huber_beta == pytest.approx(1.0 / 11.885)
    torch.testing.assert_close(torch.stack(losses), losses[0].expand(3))


def test_v1_position_loss_keeps_historical_normalized_beta() -> None:
    assert PLCSLoss(_loss_config(), normalization="v1").position_huber_beta == 1.0

    with pytest.raises(ValueError, match="not representable"):
        PLCSLoss(
            replace(_loss_config(), position_huber_beta_m=1.0),
            normalization="v1",
        )


def _tracking_config() -> SimpleNamespace:
    return SimpleNamespace(
        position_weight=1.0,
        rotation_weight=0.5,
        presence_weight=1.0,
        presence_inactive_weight=0.25,
        presence_active_weight=1.0,
        presence_transition_weight=2.0,
        transition_radius=2,
        track_smoothness_weight=0.0,
        match_position_weight=1.0,
        match_rotation_weight=0.5,
        match_presence_weight=0.5,
        position_huber_beta_m=None,
    )


def test_plcs_tracking_loss_and_hungarian_cost_share_v2_physical_beta() -> None:
    v1 = PLCSTrackingLoss(_tracking_config(), normalization="v1")
    v2 = PLCSTrackingLoss(_tracking_config(), normalization="v2")

    assert v1.position_huber_beta == 1.0
    assert v2.position_huber_beta == pytest.approx(1.0 / 11.885)
    assert not hasattr(v2, "position_axis_weights")
