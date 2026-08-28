"""Unit tests for PLCS Lightning persistence payloads."""

from __future__ import annotations

from typing import cast

import torch

from src.tasks.base.model_io import ModelCall
from src.tasks.plcs.model_io import PLCSDecodedPrediction, PLCSPreparedBatch
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.utils.geometry.court_pose import canonical_pose_to_world_pose


def test_canonical_test_payload_persists_prediction_and_physical_target() -> None:
    position = torch.tensor([[[0.25, -0.10, 0.20], [-0.15, 0.05, 0.30]]])
    rotation = torch.tensor([[[0.0, 1.0], [0.6, 0.8]]])
    canonical_target = torch.randn(1, 2, 17, 3)
    world_pose = canonical_pose_to_world_pose(
        canonical_target,
        position,
        rotation,
    )
    prediction = torch.randn_like(world_pose)
    prepared = PLCSPreparedBatch(
        call=ModelCall(),
        target_position=position,
        target_rotation=rotation,
        target_human_kp_3d=world_pose,
    )
    result = {
        "outputs": PLCSDecodedPrediction(
            position=position,
            rotation=rotation,
            canonical_pose=prediction,
        ),
        "prepared": prepared,
    }
    module = cast("PLCSLightningModule", object())

    payload = PLCSLightningModule.test_prediction_payload(module, {}, result)

    torch.testing.assert_close(payload["pred_canonical_pose"], prediction)
    torch.testing.assert_close(payload["target_canonical_pose"], canonical_target)
