from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.blcs.training.tracking_losses import BLCSTrackingLoss


def _criterion() -> BLCSTrackingLoss:
    config = OmegaConf.load(Path("src/tasks/blcs/configs/loss/tracking.yaml"))
    return BLCSTrackingLoss(config)


def _fixture():
    torch.manual_seed(4)
    prediction = {
        "position": torch.rand(1, 5, 3, 3, requires_grad=True),
        "presence_logits": torch.randn(1, 5, 3, requires_grad=True),
    }
    batch = {
        "position_3d": torch.rand(1, 5, 2, 3),
        "ball_present": torch.tensor(
            [[[1, 0], [1, 1], [1, 1], [0, 1], [0, 0]]], dtype=torch.bool
        ),
        "target_ball_mask": torch.ones(1, 2, dtype=torch.bool),
        "frame_mask": torch.ones(1, 5, dtype=torch.bool),
    }
    return prediction, batch


def test_loss_is_invariant_to_gt_ball_order() -> None:
    prediction, batch = _fixture()
    original, _ = _criterion()(prediction, batch)
    permutation = torch.tensor([1, 0])
    permuted = dict(batch)
    permuted["position_3d"] = batch["position_3d"][:, :, permutation]
    permuted["ball_present"] = batch["ball_present"][:, :, permutation]
    permuted["target_ball_mask"] = batch["target_ball_mask"][:, permutation]
    reordered, _ = _criterion()(prediction, permuted)
    torch.testing.assert_close(original["total"], reordered["total"])


def test_all_balls_absent_is_finite_and_only_presence_is_active() -> None:
    prediction, batch = _fixture()
    batch["target_ball_mask"].zero_()
    batch["ball_present"].zero_()
    losses, assignments = _criterion()(prediction, batch)
    assert torch.isfinite(losses["total"])
    assert losses["position"].item() == 0.0
    assert assignments[0][0].numel() == 0
    losses["total"].backward()


def test_matching_accepts_bfloat16_predictions() -> None:
    prediction, batch = _fixture()
    prediction = {
        key: value.detach().to(torch.bfloat16).requires_grad_()
        for key, value in prediction.items()
    }
    batch["position_3d"] = batch["position_3d"].to(torch.bfloat16)

    losses, assignments = _criterion()(prediction, batch)

    assert torch.isfinite(losses["total"])
    assert assignments[0][0].numel() == 2
