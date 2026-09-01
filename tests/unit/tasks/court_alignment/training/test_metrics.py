"""Unit tests for sigma-comparable alignment diagnostics."""

from __future__ import annotations

import torch

from src.tasks.court_alignment.training.metrics import peak_metrics


def test_peak_metrics_report_pixel_error_and_recall() -> None:
    logits = torch.full((1, 14, 20, 20), -10.0)
    logits[:, :, 10, 10] = 10.0
    keypoints = torch.tensor([[[[10.0, 10.0]] * 14]])
    visibility = torch.ones(1, 14, 1, dtype=torch.bool)

    result = peak_metrics(logits, keypoints, visibility, image_size=(20, 20), target_normalized=False)

    assert result["peak_mean_error_px"] == 0.0
    assert result["recall_at_2px"] == 1.0
    assert result["recall_at_4px"] == 1.0
