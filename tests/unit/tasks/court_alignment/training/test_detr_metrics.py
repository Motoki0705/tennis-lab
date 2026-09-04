"""Unit tests for DINO oriented-court metrics."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.court_alignment.inference.detr_decoder import decode_detr_courts
from src.tasks.court_alignment.training.detr_metrics import CourtDetrMetrics


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _target(*court_boxes: tuple[float, float, float, float, float]) -> dict[str, torch.Tensor]:
    values = torch.tensor(court_boxes, dtype=torch.float32).reshape(-1, 5)
    count = values.shape[0]
    return {
        "labels": torch.zeros(count, dtype=torch.long),
        "boxes": torch.zeros((count, 4), dtype=torch.float32),
        "court_boxes": values,
    }


def test_exact_multi_instance_predictions_have_zero_pose_errors() -> None:
    logits = torch.tensor([[[8.0], [7.0], [-8.0]]])
    boxes = torch.tensor(
        [[[0.25, 0.4, 0.2, 0.3], [0.75, 0.6, 0.3, 0.2], [0.5, 0.5, 0.1, 0.1]]]
    )
    court = torch.tensor(
        [[[_logit(0.3), 1.0, 0.0], [_logit(0.4), 0.0, 1.0], [0.0, 1.0, 0.0]]]
    )
    predictions = decode_detr_courts(
        logits,
        boxes,
        court,
        image_size=800,
        threshold=0.5,
        top_k=3,
    )
    targets = [_target((0.25, 0.4, 0.3, 1.0, 0.0), (0.75, 0.6, 0.4, 0.0, 1.0))]
    metrics = CourtDetrMetrics(match_max_corner_error_px=1.0)

    metrics.update(predictions, targets, image_size=800)
    result = metrics.compute()

    assert result["instance_tp"] == 2.0
    assert result["instance_fp"] == 0.0
    assert result["instance_fn"] == 0.0
    assert result["instance_f1"] == 1.0
    assert result["instance_count_accuracy"] == 1.0
    assert result["matched_center_mean_error_px"] == pytest.approx(0.0)
    assert result["matched_scale_relative_error"] == pytest.approx(0.0)
    assert result["matched_axial_angle_mean_error_deg"] == pytest.approx(0.0)
    assert result["matched_corner_mean_error_px"] == pytest.approx(0.0)


def test_empty_batch_uses_finite_count_and_pose_conventions() -> None:
    predictions = decode_detr_courts(
        torch.full((1, 2, 1), -8.0),
        torch.full((1, 2, 4), 0.5),
        torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]),
        image_size=(600, 800),
        threshold=0.5,
        top_k=2,
    )
    metrics = CourtDetrMetrics()

    metrics.update(predictions, [_target()], image_size=(600, 800))
    result = metrics.compute()

    assert result["instance_count_accuracy"] == 1.0
    assert result["instance_count_mae"] == 0.0
    assert result["instance_precision"] == 0.0
    assert result["instance_recall"] == 0.0
    assert result["matched_center_mean_error_px"] == pytest.approx(
        math.hypot(799.0, 599.0)
    )
    assert result["matched_axial_angle_mean_error_deg"] == 90.0
    assert all(math.isfinite(value) for value in result.values())


def test_geometrically_wrong_assignment_remains_fp_and_fn() -> None:
    predictions = decode_detr_courts(
        torch.tensor([[[8.0]]]),
        torch.tensor([[[0.1, 0.1, 0.2, 0.2]]]),
        torch.tensor([[[_logit(0.2), 1.0, 0.0]]]),
        image_size=800,
        threshold=0.5,
        top_k=1,
    )
    metrics = CourtDetrMetrics(match_max_corner_error_px=8.0)

    metrics.update(
        predictions,
        [_target((0.9, 0.9, 0.2, 1.0, 0.0))],
        image_size=800,
    )
    result = metrics.compute()

    assert result["instance_tp"] == 0.0
    assert result["instance_fp"] == 1.0
    assert result["instance_fn"] == 1.0
    assert result["instance_f1"] == 0.0
