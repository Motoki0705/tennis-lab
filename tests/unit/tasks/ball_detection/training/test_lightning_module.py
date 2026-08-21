"""Unit tests for ball detection Lightning test-prediction payloads."""

from __future__ import annotations

import pytest
import torch

from src.tasks.ball_detection.model_io.contracts import (
    BallModelCall,
    BallTrainingCall,
)
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)

pytestmark = pytest.mark.unit


def test_gt_trajectory_marks_frames_without_a_visible_ball_as_padding() -> None:
    module = object.__new__(BallDetectionLightningModule)
    coords = torch.tensor(
        [
            [
                [[1.0, 2.0], [5.0, 10.0]],
                [[3.0, 4.0], [7.0, 12.0]],
                [[9.0, 16.0], [2.0, 6.0]],
            ]
        ]
    )
    visibility = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]])
    images = torch.zeros(1, 3, 3, 2, 2)
    call = BallTrainingCall(
        model_call=BallModelCall(
            images=images,
            model_input=images,
            model_args=(images,),
            batch_size=1,
            frame_count=3,
        ),
        target_heatmaps=torch.zeros(1, 3, 2, 2),
        coords=coords,
        visibility=visibility,
        original_size=torch.tensor([[11.0, 21.0]]),
    )

    trajectory, padding_mask = module._extract_gt_trajectory(call)

    expected_trajectory = torch.tensor([[[0.5, 0.5], [0.3, 0.2], [0.9, 0.8]]])
    torch.testing.assert_close(trajectory, expected_trajectory)
    assert torch.equal(padding_mask, torch.tensor([[False, True, False]]))


def test_test_prediction_payload_persists_tracknet_predictions_and_targets() -> None:
    module = object.__new__(BallDetectionLightningModule)
    pred_heatmaps = torch.rand(2, 3, 4, 5)
    batch = {
        "coords": torch.rand(2, 3, 4, 2),
        "visibility": torch.ones(2, 3, 4),
        "original_size": torch.tensor([[1280.0, 720.0], [1280.0, 720.0]]),
        "heatmap_size": torch.tensor([[5.0, 4.0], [5.0, 4.0]]),
        "window_id": ["Game1/Clip1:0", "Game1/Clip1:4"],
    }

    payload = module.test_prediction_payload(
        batch,
        {"pred_heatmaps": pred_heatmaps},
    )

    assert payload["pred_heatmaps"] is pred_heatmaps
    assert payload["window_id"] == ["Game1/Clip1:0", "Game1/Clip1:4"]
    assert payload["target_coords"] is batch["coords"]
    assert payload["target_visibility"] is batch["visibility"]
    assert payload["original_size"] is batch["original_size"]
    assert payload["heatmap_size"] is batch["heatmap_size"]
