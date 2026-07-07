"""Tests for court_detection metrics."""

from __future__ import annotations

import torch

from src.tasks.court_detection.training.metrics import CourtDetectionMetrics


def _logits_with_peaks(coords: torch.Tensor, h: int = 16, w: int = 16) -> torch.Tensor:
    """Return logits whose per-channel argmax sits at integer ``(x, y)``."""
    batch_size, num_keypoints, _ = coords.shape
    logits = torch.zeros(batch_size, num_keypoints, h, w)
    for batch_idx in range(batch_size):
        for kp_idx in range(num_keypoints):
            x, y = int(coords[batch_idx, kp_idx, 0]), int(coords[batch_idx, kp_idx, 1])
            logits[batch_idx, kp_idx, y, x] = 10.0
    return logits


def test_kp_metric_ignores_invisible_keypoints() -> None:
    metrics = CourtDetectionMetrics("kp", {"num_keypoints": 2})
    gt = torch.tensor([[[4.0, 4.0], [10.0, 10.0]]])
    pred = torch.tensor([[[4.0, 4.0], [0.0, 0.0]]])
    batch = {
        "keypoints": gt,
        "kp_visible": torch.tensor([[True, False]]),
    }

    metrics.update(_logits_with_peaks(pred), batch)

    assert metrics.compute()["mean_dist"] == 0.0


def test_kp_metric_counts_visible_error() -> None:
    metrics = CourtDetectionMetrics("kp", {"num_keypoints": 2})
    gt = torch.tensor([[[4.0, 4.0], [10.0, 10.0]]])
    pred = torch.tensor([[[7.0, 8.0], [10.0, 10.0]]])
    batch = {
        "keypoints": gt,
        "kp_visible": torch.tensor([[True, True]]),
    }

    metrics.update(_logits_with_peaks(pred), batch)

    assert metrics.compute()["mean_dist"] == 2.5


def test_kp_metric_all_invisible_is_zero() -> None:
    metrics = CourtDetectionMetrics("kp", {"num_keypoints": 1})
    batch = {
        "keypoints": torch.tensor([[[4.0, 4.0]]]),
        "kp_visible": torch.tensor([[False]]),
    }

    metrics.update(_logits_with_peaks(torch.tensor([[[0.0, 0.0]]])), batch)

    assert metrics.compute()["mean_dist"] == 0.0
