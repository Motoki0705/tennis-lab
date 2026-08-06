"""Unit tests for court detection Lightning test-prediction payloads."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.model_io.adapters import (
    CourtKeypointModelIO,
    CourtLineModelIO,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import CourtModelSpec, CourtTask
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)

pytestmark = pytest.mark.unit


def _module_for_task(
    task: CourtTask,
    *,
    output_channels: int,
) -> CourtDetectionLightningModule:
    module = object.__new__(CourtDetectionLightningModule)
    torch.nn.Module.__init__(module)
    spec = CourtModelSpec(
        task=task,
        in_channels=3,
        output_channels=output_channels,
        short_side=32,
    )
    if task == "kp":
        module.model_io = CourtKeypointModelIO(spec, focal_gamma=2.0)
    elif task == "seg":
        module.model_io = CourtSegmentationModelIO(
            spec,
            ce_weight=1.0,
            dice_weight=1.0,
        )
    else:
        module.model_io = CourtLineModelIO(
            spec,
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=1.0,
        )
    return module


def test_kp_test_prediction_payload_saves_predicted_and_target_keypoints() -> None:
    module = _module_for_task("kp", output_channels=3)
    logits = torch.zeros(2, 3, 4, 5)
    logits[0, :, 1, 2] = 10.0
    logits[1, :, 3, 4] = 10.0
    batch = {
        "keypoints": torch.rand(2, 3, 2),
        "image_size": torch.tensor([[4, 5], [4, 5]]),
        "image_id": ["a", "b"],
    }

    payload = module.test_prediction_payload(batch, {"logits": logits})

    assert payload["image_id"] == ["a", "b"]
    assert payload["image_size"] is batch["image_size"]
    assert payload["pred_keypoints"].shape == (2, 3, 2)
    assert payload["target_keypoints"] is batch["keypoints"]
    assert torch.all(payload["pred_keypoints"][0] == torch.tensor([2.0, 1.0]))
    assert torch.all(payload["pred_keypoints"][1] == torch.tensor([4.0, 3.0]))


def test_seg_test_prediction_payload_flattens_variable_spatial_masks() -> None:
    module = _module_for_task("seg", output_channels=4)
    logits = torch.zeros(2, 4, 2, 3)
    logits[:, 2] = 1.0
    target = torch.ones(2, 2, 3, dtype=torch.long)
    batch = {
        "mask": target,
        "image_size": torch.tensor([[2, 3], [2, 3]]),
        "image_id": ["a", "b"],
    }

    payload = module.test_prediction_payload(batch, {"logits": logits})

    assert payload["pred_mask_flat"].shape == (2, 6)
    assert payload["target_mask_flat"].shape == (2, 6)
    assert payload["padded_size"].tolist() == [[2, 3], [2, 3]]
    assert torch.all(payload["pred_mask_flat"] == 2)


def test_line_test_prediction_payload_flattens_probabilities_and_targets() -> None:
    module = _module_for_task("line", output_channels=1)
    logits = torch.zeros(2, 1, 2, 3)
    target = torch.ones(2, 1, 2, 3)
    batch = {
        "mask": target,
        "image_size": torch.tensor([[2, 3], [2, 3]]),
        "image_id": ["a", "b"],
    }

    payload = module.test_prediction_payload(batch, {"logits": logits})

    assert payload["pred_line_prob_flat"].shape == (2, 6)
    assert payload["target_line_mask_flat"].shape == (2, 6)
    assert payload["padded_size"].tolist() == [[2, 3], [2, 3]]
    assert torch.allclose(payload["pred_line_prob_flat"], torch.full((2, 6), 0.5))
