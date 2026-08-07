from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import src.tasks.blcs.training.tracking_losses as tracking_losses_module
from src.tasks.base.model_io import ModelCall
from src.tasks.blcs.model_io import (
    BLCSTrackQueryPrediction,
    BLCSTrackQueryTrainingBatch,
)
from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss


def _criterion() -> BLCSTrackingLoss:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/loss/tracking.yaml"))
    return BLCSTrackingLoss(config)


def _prediction(position: torch.Tensor, logits: torch.Tensor) -> BLCSTrackQueryPrediction:
    probability = logits.sigmoid()
    return BLCSTrackQueryPrediction(
        position=position,
        presence_logits=logits,
        presence_probability=probability,
        presence=probability >= 0.5,
    )


def _fixture() -> tuple[BLCSTrackQueryPrediction, BLCSTrackQueryTrainingBatch]:
    torch.manual_seed(4)
    position = torch.rand(1, 5, 3, 3, requires_grad=True)
    logits = torch.randn(1, 5, 3, requires_grad=True)
    target_position = torch.rand(1, 5, 2, 3)
    target_presence = torch.tensor(
            [[[1, 0], [1, 1], [1, 1], [0, 1], [0, 0]]], dtype=torch.bool
        )
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=target_position,
        target_velocity=torch.zeros_like(target_position),
        target_presence=target_presence,
        target_instance_id=torch.where(
            target_presence,
            torch.arange(2).view(1, 1, 2).expand(1, 5, 2),
            -1,
        ),
        target_slot_mask=torch.ones(1, 2, dtype=torch.bool),
        frame_mask=torch.ones(1, 5, dtype=torch.bool),
    )
    prediction = _prediction(position, logits)
    return prediction, batch


def _compute(
    criterion: BLCSTrackingLoss,
    prediction: BLCSTrackQueryPrediction,
    batch: BLCSTrackQueryTrainingBatch,
) -> tuple[dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    inputs, assignments = criterion.prepare_inputs(prediction, batch)
    return criterion(inputs), assignments


def test_loss_is_invariant_to_gt_ball_order() -> None:
    prediction, batch = _fixture()
    original, _ = _compute(_criterion(), prediction, batch)
    permutation = torch.tensor([1, 0])
    permuted = replace(
        batch,
        target_position=batch.target_position[:, :, permutation],
        target_velocity=batch.target_velocity[:, :, permutation],
        target_presence=batch.target_presence[:, :, permutation],
        target_instance_id=batch.target_instance_id[:, :, permutation],
        target_slot_mask=batch.target_slot_mask[:, permutation],
    )
    reordered, _ = _compute(_criterion(), prediction, permuted)
    torch.testing.assert_close(original["total"], reordered["total"])


def test_all_balls_absent_is_finite_and_only_presence_is_active() -> None:
    prediction, batch = _fixture()
    batch.target_slot_mask.zero_()
    batch.target_presence.zero_()
    losses, assignments = _compute(_criterion(), prediction, batch)
    assert torch.isfinite(losses["total"])
    assert losses["position"].item() == 0.0
    assert assignments[0][0].numel() == 0
    losses["total"].backward()


def test_matching_accepts_bfloat16_predictions() -> None:
    prediction, batch = _fixture()
    prediction = _prediction(
        prediction.position.detach().to(torch.bfloat16).requires_grad_(),
        prediction.presence_logits.detach().to(torch.bfloat16).requires_grad_(),
    )
    batch = replace(
        batch,
        target_position=batch.target_position.to(torch.bfloat16),
        target_velocity=batch.target_velocity.to(torch.bfloat16),
    )

    losses, assignments = _compute(_criterion(), prediction, batch)

    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 2


def test_forward_only_combines_boundary_prepared_tensor_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction, batch = _fixture()
    criterion = _criterion()
    inputs, _ = criterion.prepare_inputs(prediction, batch)

    def _unexpected_matching(*args: object, **kwargs: object) -> None:
        raise AssertionError("matching must not run from forward")

    monkeypatch.setattr(tracking_losses_module, "match_ball_tracks", _unexpected_matching)

    losses = criterion(inputs)

    assert torch.isfinite(losses["total"])


def test_position_loss_reports_axes_and_uses_configured_balance() -> None:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/loss/tracking.yaml"))
    config.presence_weight = 0.0
    criterion = BLCSTrackingLoss(config)
    prediction = _prediction(
        torch.tensor([[[[1.0, 2.0, 3.0]]]], requires_grad=True),
        torch.tensor([[[20.0]]], requires_grad=True),
    )
    batch = BLCSTrackQueryTrainingBatch(
        call=ModelCall(kwargs={}),
        target_position=torch.zeros(1, 1, 1, 3),
        target_velocity=torch.zeros(1, 1, 1, 3),
        target_presence=torch.ones(1, 1, 1, dtype=torch.bool),
        target_instance_id=torch.zeros(1, 1, 1, dtype=torch.long),
        target_slot_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    losses, _ = _compute(criterion, prediction, batch)

    torch.testing.assert_close(losses["position_x"], torch.tensor(0.5))
    torch.testing.assert_close(losses["position_y"], torch.tensor(1.5))
    torch.testing.assert_close(losses["position_z"], torch.tensor(2.5))
    torch.testing.assert_close(losses["position"], torch.tensor(1.3))


def test_tracking_loss_rejects_invalid_position_axis_weights() -> None:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/loss/tracking.yaml"))
    config.position_axis_weights = [1.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="finite and positive"):
        BLCSTrackingLoss(config)
