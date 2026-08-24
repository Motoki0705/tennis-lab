"""Tests for bundle-aware Court metrics."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.geometry.pose import CourtDecodedPose
from src.tasks.court_detection.model_io.contracts import (
    CourtPoseTargetBatch,
    CourtQueryConsistencyResult,
)
from src.tasks.court_detection.training.metrics import (
    CourtDetectionMetrics,
    CourtPoseMetrics,
    CourtQueryGeometryMetrics,
    gradient_finite_status,
)


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
    metrics = CourtDetectionMetrics("kp", 1, singleton_kp=True)
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

    assert metrics.compute() == {
        "mean_dist": 0.0,
        "mean_distance_px": 0.0,
        "median_distance_px": 0.0,
    }


def test_singleton_metric_uses_one_global_peak_not_nearest_fallback() -> None:
    metrics = CourtDetectionMetrics("kp", 1, singleton_kp=True)
    expected = torch.tensor([[[[4.0, 4.0]]]])
    target = _target(expected, visible=torch.tensor([[[True]]]))
    logits = torch.full((1, 1, 16, 16), -10.0)
    logits[0, 0, 4, 4] = 9.0  # Correct, but not the singleton prediction.
    logits[0, 0, 12, 12] = 10.0

    metrics.update(
        logits,
        target,
        image_size=torch.tensor([[16, 16]], dtype=torch.long),
    )

    assert metrics.compute()["mean_dist"] == pytest.approx(8.0 * 2.0**0.5)


def test_singleton_metric_excludes_padding_and_reports_median_pixels() -> None:
    metrics = CourtDetectionMetrics("kp", 2, singleton_kp=True)
    expected = torch.tensor([[[[2.0, 1.0]], [[6.0, 3.0]]]])
    target = _target(
        expected,
        visible=torch.ones(1, 2, 1, dtype=torch.bool),
        height=4,
        width=7,
    )
    logits = torch.full((1, 2, 8, 10), -10.0)
    logits[0, 0, 1, 2] = 9.0
    logits[0, 1, 3, 6] = 9.0
    logits[:, :, 7, 9] = 100.0

    metrics.update(
        logits,
        target,
        image_size=torch.tensor([[4, 7]], dtype=torch.long),
    )

    assert metrics.compute() == {
        "mean_dist": 0.0,
        "mean_distance_px": 0.0,
        "median_distance_px": 0.0,
    }


def test_pose_metrics_report_metric_translation_rotation_and_focal() -> None:
    metrics = CourtPoseMetrics()
    prediction = CourtDecodedPose(
        translation_m=torch.tensor([[1.0, 2.0, 3.0]]),
        rotation=torch.eye(3).unsqueeze(0),
        focal_px=torch.tensor([200.0]),
        log_focal=torch.log(torch.tensor([200.0])),
    )
    target = CourtPoseTargetBatch(
        translation_m=torch.tensor([[1.0, 2.0, 3.0]]),
        rotation=torch.eye(3).unsqueeze(0),
        log_focal=torch.log(torch.tensor([200.0])),
        intrinsics=torch.eye(3).unsqueeze(0),
        semantic_to_physical=torch.arange(14).unsqueeze(0),
        raw_pose10d=torch.zeros(1, 10),
    )

    metrics.update(prediction, target)

    assert metrics.compute() == pytest.approx({
        "translation_l2_m": 0.0,
        "rotation_geodesic_deg": 0.0,
        "focal_relative_error": 0.0,
        "log_focal_abs_error": 0.0,
    }, abs=1.0e-6)


def test_query_geometry_metrics_report_pose_only_consistency_and_depth() -> None:
    tracker = CourtQueryGeometryMetrics(min_depth_m=0.1)
    ground_truth = torch.zeros(1, 14, 2)
    ground_truth[:, :, 0] = 0.5
    ground_truth[:, :, 1] = 0.5
    pose = torch.tensor([[[5.0, 4.0]]]).expand(1, 14, 2).clone()
    dense = pose.clone()
    dense[:, :, 0] += 2.0
    visible = torch.zeros(1, 14, dtype=torch.bool)
    visible[:, :2] = True
    depth = torch.ones(1, 14)
    depth[:, 0] = 0.0
    consistency = CourtQueryConsistencyResult(
        coordinate_loss=torch.tensor(0.0),
        cheirality_loss=torch.tensor(0.0),
        auxiliary_loss=torch.tensor(0.0),
        weighted_auxiliary_loss=torch.tensor(0.0),
        effective_weight=torch.tensor(1.0),
        visible_point_count=torch.tensor(2),
        mean_distance_px=torch.tensor(2.0),
        invalid_depth_rate=torch.tensor(0.5),
        dense_points_xy=dense,
        pose_points_xy=pose,
        pose_depth_m=depth,
    )

    tracker.update(
        consistency,
        ground_truth_points_normalized=ground_truth,
        point_visible=visible,
        image_size=torch.tensor([[9, 11]], dtype=torch.long),
    )

    assert tracker.compute() == pytest.approx(
        {
            "pose_reprojection_mean_distance_px": 0.0,
            "kp_pose_consistency_distance_px": 2.0,
            "invalid_depth_rate": 0.5,
            "visible_point_count": 2.0,
        }
    )


def test_gradient_finite_status_requires_present_finite_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    assert gradient_finite_status([parameter]) == 0.0

    parameter.grad = torch.tensor([2.0])
    assert gradient_finite_status([parameter]) == 1.0

    parameter.grad = torch.tensor([float("nan")])
    assert gradient_finite_status([parameter]) == 0.0
