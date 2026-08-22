"""Tests for bundle-aware Court metrics."""

from __future__ import annotations

import torch

from src.tasks.court_detection.training.metrics import CourtDetectionMetrics


def _logits_with_peaks(
    coords: torch.Tensor,
    *,
    height: int = 16,
    width: int = 16,
) -> torch.Tensor:
    """Return logits with one or more explicit peaks per channel."""
    batch_size, channels, points, _ = coords.shape
    logits = torch.full((batch_size, channels, height, width), -10.0)
    for batch_index in range(batch_size):
        for channel_index in range(channels):
            for point_index in range(points):
                x = int(coords[batch_index, channel_index, point_index, 0])
                y = int(coords[batch_index, channel_index, point_index, 1])
                logits[batch_index, channel_index, y, x] = 10.0
    return logits


def _target(
    points_px: torch.Tensor,
    *,
    visible: torch.Tensor,
    height: int = 16,
    width: int = 16,
) -> dict[str, torch.Tensor]:
    scale = points_px.new_tensor([width - 1.0, height - 1.0])
    batch_size, channels, points, _ = points_px.shape
    return {
        "heatmap": torch.zeros(batch_size, channels, height, width),
        "points_xy": points_px / scale,
        "point_visible": visible,
        "physical_indices": torch.arange(points)
        .view(1, 1, points)
        .expand(batch_size, channels, points)
        .long(),
    }


def test_kp_metric_ignores_invisible_points() -> None:
    metrics = CourtDetectionMetrics("kp", 2)
    expected = torch.tensor([[[[4.0, 4.0]], [[10.0, 10.0]]]])
    predicted = torch.tensor([[[[4.0, 4.0]], [[0.0, 0.0]]]])
    target = _target(
        expected,
        visible=torch.tensor([[[True], [False]]]),
    )

    metrics.update(
        _logits_with_peaks(predicted),
        target,
        image_size=torch.tensor([[16, 16]], dtype=torch.long),
    )

    assert metrics.compute()["mean_dist"] == 0.0


def test_kp_metric_matches_each_visible_point_to_nearest_peak() -> None:
    metrics = CourtDetectionMetrics("kp", 1)
    expected = torch.tensor([[[[4.0, 4.0], [10.0, 10.0]]]])
    predicted = torch.tensor([[[[7.0, 8.0], [10.0, 10.0]]]])
    target = _target(
        expected,
        visible=torch.tensor([[[True, True]]]),
    )

    metrics.update(
        _logits_with_peaks(predicted),
        target,
        image_size=torch.tensor([[16, 16]], dtype=torch.long),
    )

    assert metrics.compute()["mean_dist"] == 2.5


def test_kp_metric_all_invisible_is_zero() -> None:
    metrics = CourtDetectionMetrics("kp", 1)
    expected = torch.tensor([[[[4.0, 4.0]]]])
    target = _target(
        expected,
        visible=torch.tensor([[[False]]]),
    )

    metrics.update(
        _logits_with_peaks(torch.tensor([[[[0.0, 0.0]]]])),
        target,
        image_size=torch.tensor([[16, 16]], dtype=torch.long),
    )

    assert metrics.compute()["mean_dist"] == 0.0


def test_target_court_metric_uses_single_point_capacity() -> None:
    metrics = CourtDetectionMetrics("kp", 1)
    expected = torch.tensor([[[[4.0, 4.0]]]])
    target = _target(expected, visible=torch.tensor([[[True]]]))
    logits = torch.full((1, 1, 16, 16), -10.0)
    logits[0, 0, 4, 4] = 10.0
    logits[0, 0, 12, 12] = 9.0  # A non-target-court-like second peak.

    metrics.update(
        logits,
        target,
        image_size=torch.tensor([[16, 16]], dtype=torch.long),
    )

    assert metrics.compute()["mean_dist"] == 0.0
