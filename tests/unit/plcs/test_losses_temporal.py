from __future__ import annotations

import math

import pytest
import torch

from src.plcs.training.losses import PLCSLoss, PLCSLossConfig, TemporalTermConfig, TemporalTermsConfig


@pytest.mark.unit
def test_loss_masks_padded_frames() -> None:
    loss_fn = PLCSLoss(
        config=PLCSLossConfig(
            position_weight=1.0,
            rotation_weight=1.0,
            temporal_terms=TemporalTermsConfig(
                position_gt=TemporalTermConfig(weight=0.0),
                position_inertia=TemporalTermConfig(weight=0.0),
                rotation_gt=TemporalTermConfig(weight=0.0),
                rotation_inertia=TemporalTermConfig(weight=0.0),
            ),
        )
    )

    # Two valid frames + one padded frame.
    pred_pos = torch.tensor([[[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [9.9, 9.9, 9.9]]])
    tgt_pos = torch.tensor([[[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.0, 0.0, 0.0]]])

    pred_rot = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])
    tgt_rot = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]]])

    seq_mask = torch.tensor([[True, True, False]])

    losses = loss_fn(
        pred_position=pred_pos,
        pred_rotation=pred_rot,
        target_position=tgt_pos,
        target_rotation=tgt_rot,
        seq_mask=seq_mask,
    )

    assert losses["position"].item() == pytest.approx(0.0)
    assert losses["rotation"].item() == pytest.approx(0.0)


@pytest.mark.unit
def test_rotation_temporal_gt_handles_wraparound() -> None:
    cfg = PLCSLossConfig(
        position_weight=0.0,
        rotation_weight=0.0,
        temporal_terms=TemporalTermsConfig(
            rotation_gt=TemporalTermConfig(weight=1.0, order=1, robust=True)
        ),
    )
    loss_fn = PLCSLoss(config=cfg)

    # Yaw crosses the -pi/pi boundary. Velocity should be small, not ~2*pi.
    yaw0 = math.pi - 0.02
    yaw1 = -math.pi + 0.02
    target_yaw = torch.tensor([[yaw0, yaw1]])
    pred_yaw = target_yaw.clone()

    target_rot = torch.stack([torch.sin(target_yaw), torch.cos(target_yaw)], dim=-1)  # (B, T, 2)
    pred_rot = torch.stack([torch.sin(pred_yaw), torch.cos(pred_yaw)], dim=-1)

    dummy_pos = torch.zeros((1, 2, 3))
    losses = loss_fn(
        pred_position=dummy_pos,
        pred_rotation=pred_rot,
        target_position=dummy_pos,
        target_rotation=target_rot,
        seq_mask=torch.tensor([[True, True]]),
    )

    assert losses["rotation_temporal_gt"].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_position_inertia_order2_is_zero_for_constant_velocity() -> None:
    cfg = PLCSLossConfig(
        position_weight=0.0,
        rotation_weight=0.0,
        temporal_terms=TemporalTermsConfig(
            position_inertia=TemporalTermConfig(weight=1.0, order=2, robust=False)
        ),
    )
    loss_fn = PLCSLoss(config=cfg)

    # Constant velocity in X: 0, 1, 2, 3 -> zero acceleration.
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    rot = torch.zeros((1, 4, 2))
    mask = torch.tensor([[True, True, True, True]])

    losses = loss_fn(
        pred_position=pos,
        pred_rotation=rot,
        target_position=pos,
        target_rotation=rot,
        seq_mask=mask,
    )

    assert losses["position_temporal_inertia"].item() == pytest.approx(0.0, abs=1e-8)

