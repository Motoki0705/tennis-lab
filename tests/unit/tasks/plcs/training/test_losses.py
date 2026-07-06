"""Unit tests for the PLCS position-smoothness loss term."""

from __future__ import annotations

import torch

from src.tasks.plcs.training.losses import (
    DEFAULT_LOSS_TERMS,
    PLCSLoss,
    PLCSLossConfig,
    PLCSLossInputs,
    position_smoothness_loss_term,
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


def test_registered_and_off_by_default() -> None:
    assert "position_smoothness" in DEFAULT_LOSS_TERMS
    assert PLCSLossConfig().position_smoothness_weight == 0.0
    assert PLCSLoss().weight_for("position_smoothness") == 0.0


def test_from_dict_parses_weight() -> None:
    cfg = PLCSLossConfig.from_dict({"position_smoothness_weight": 4.0})
    assert cfg.position_smoothness_weight == 4.0


def test_frame_level_input_is_noop() -> None:
    # (B, 3): no temporal axis -> term must be a no-op, not misread coords as time.
    pred = torch.randn(5, 3)
    assert position_smoothness_loss_term(_inputs(pred, None)).item() == 0.0


def test_jitter_penalized_more_than_smooth() -> None:
    torch.manual_seed(0)
    t = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
    mask = torch.ones(1, 40)
    smooth = position_smoothness_loss_term(_inputs(t, mask))
    jittery = position_smoothness_loss_term(_inputs(t + 0.02 * torch.randn(1, 40, 3), mask))
    assert jittery > 10 * smooth


def test_contributes_to_total_via_forward() -> None:
    torch.manual_seed(1)
    cfg = PLCSLossConfig(
        position_weight=0.0, rotation_weight=0.0, position_smoothness_weight=4.0
    )
    loss_fn = PLCSLoss(config=cfg)
    pred_pos = torch.randn(1, 20, 3)
    pred_rot = torch.randn(1, 20, 2)
    losses = loss_fn(
        pred_position=pred_pos,
        pred_rotation=pred_rot,
        target_position=pred_pos,
        target_rotation=pred_rot,
        human_mask=torch.ones(1, 20),
    )
    assert losses["position_smoothness"] > 0
    torch.testing.assert_close(losses["total"], 4.0 * losses["position_smoothness"])
