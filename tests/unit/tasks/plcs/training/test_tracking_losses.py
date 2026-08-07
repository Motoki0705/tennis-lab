from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.plcs.training.tracking_losses import PLCSTrackingLoss


def _criterion() -> PLCSTrackingLoss:
    config = OmegaConf.load(Path("src/tasks/plcs/configs/loss/tracking.yaml"))
    return PLCSTrackingLoss(config)


def _fixture():
    torch.manual_seed(8)
    prediction = {
        "position": torch.rand(1, 5, 3, 3, requires_grad=True),
        "rotation": torch.nn.functional.normalize(
            torch.rand(1, 5, 3, 2), dim=-1
        ).requires_grad_(),
        "presence_logits": torch.randn(1, 5, 3, requires_grad=True),
    }
    batch = {
        "target_position": torch.rand(1, 5, 2, 3),
        "target_rotation": torch.nn.functional.normalize(
            torch.rand(1, 5, 2, 2), dim=-1
        ),
        "target_presence": torch.tensor(
            [[[1, 0], [1, 1], [1, 1], [0, 1], [0, 0]]], dtype=torch.bool
        ),
        "target_slot_mask": torch.ones(1, 2, dtype=torch.bool),
        "frame_mask": torch.ones(1, 5, dtype=torch.bool),
    }
    return prediction, batch


def _compute(
    criterion: PLCSTrackingLoss,
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
):
    inputs, assignments = criterion.prepare_inputs(prediction, batch)
    return criterion(inputs), assignments


def test_player_loss_is_invariant_to_gt_person_order_and_smoothness_is_off() -> None:
    prediction, batch = _fixture()
    original, _ = _compute(_criterion(), prediction, batch)
    permutation = torch.tensor([1, 0])
    permuted = dict(batch)
    for key in ("target_position", "target_rotation", "target_presence"):
        permuted[key] = batch[key][:, :, permutation]
    permuted["target_slot_mask"] = batch["target_slot_mask"][:, permutation]
    reordered, _ = _compute(_criterion(), prediction, permuted)
    torch.testing.assert_close(original["total"], reordered["total"])
    assert original["track_smoothness"].item() == 0.0


def test_all_persons_absent_does_not_produce_nan() -> None:
    prediction, batch = _fixture()
    batch["target_slot_mask"].zero_()
    batch["target_presence"].zero_()
    losses, assignments = _compute(_criterion(), prediction, batch)
    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 0
    losses["total"].backward()


def test_matching_accepts_bfloat16_predictions() -> None:
    prediction, batch = _fixture()
    prediction = {
        key: value.detach().to(torch.bfloat16).requires_grad_()
        for key, value in prediction.items()
    }
    batch["target_position"] = batch["target_position"].to(torch.bfloat16)
    batch["target_rotation"] = batch["target_rotation"].to(torch.bfloat16)

    losses, assignments = _compute(_criterion(), prediction, batch)

    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 2
