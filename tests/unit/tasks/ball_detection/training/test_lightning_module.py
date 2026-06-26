"""Unit tests for ball detection Lightning test-prediction payloads."""

from __future__ import annotations

import pytest
import torch

from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)

pytestmark = pytest.mark.unit


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
