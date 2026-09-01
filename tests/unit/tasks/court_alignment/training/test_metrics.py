"""Unit tests for sigma-comparable alignment diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.court_alignment.geometry.court import canonical_court_keypoints
from src.tasks.court_alignment.inference.decoder import (
    CourtInstanceBatch,
    CourtInstances,
)
from src.tasks.court_alignment.training.metrics import (
    CourtAlignmentMetrics,
    instance_alignment_metrics,
    peak_metrics,
)
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX


def _transform(
    *,
    center: tuple[float, float],
    rotation: float = 0.0,
    scale: float = 2.0,
) -> torch.Tensor:
    canonical = canonical_court_keypoints()
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    matrix = canonical.new_tensor(((cosine, -sine), (sine, cosine)))
    return canonical @ matrix.T * scale + canonical.new_tensor(center)


def _instances(points: list[torch.Tensor], centers: list[tuple[float, float]]) -> CourtInstances:
    if points:
        keypoints = torch.stack(points)
        center_tensor = keypoints.new_tensor(centers)
    else:
        keypoints = torch.zeros((0, 14, 2))
        center_tensor = torch.zeros((0, 2))
    count = len(points)
    return CourtInstances(
        (
            CourtInstanceBatch(
                keypoints_px=keypoints,
                scores=torch.ones((count, 14)),
                valid=torch.ones((count, 14), dtype=torch.bool),
                centers_px=center_tensor,
            ),
        )
    )


def _targets(
    points: list[torch.Tensor], centers: list[tuple[float, float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if points:
        keypoints = torch.stack(points).unsqueeze(0)
        center_tensor = keypoints.new_tensor(centers).unsqueeze(0)
    else:
        keypoints = torch.zeros((1, 1, 14, 2))
        center_tensor = torch.zeros((1, 1, 2))
    visibility = torch.zeros(keypoints.shape[:3], dtype=torch.bool)
    visibility[:, : len(points)] = True
    return (
        keypoints,
        visibility,
        center_tensor,
        torch.tensor([len(points)], dtype=torch.long),
    )


def test_peak_metrics_report_pixel_error_and_recall() -> None:
    logits = torch.full((1, 14, 20, 20), -10.0)
    logits[:, :, 10, 10] = 10.0
    keypoints = torch.tensor([[[[10.0, 10.0]] * 14]])
    visibility = torch.ones(1, 14, 1, dtype=torch.bool)

    result = peak_metrics(logits, keypoints, visibility, image_size=(20, 20), target_normalized=False)

    assert result["peak_mean_error_px"] == 0.0
    assert result["recall_at_2px"] == 1.0
    assert result["recall_at_4px"] == 1.0


def test_hungarian_matching_does_not_reuse_one_prediction() -> None:
    first = _transform(center=(50.0, 50.0))
    second = _transform(center=(54.0, 50.0))
    predictions = _instances([first], [(50.0, 50.0)])
    keypoints, visibility, centers, counts = _targets(
        [first, second], [(50.0, 50.0), (54.0, 50.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["instance_tp"] == 1.0
    assert result["instance_fp"] == 0.0
    assert result["instance_fn"] == 1.0
    assert result["instance_recall"] == 0.5
    assert result["matched_instance_count"] == 1.0


def test_half_turn_semantics_are_treated_as_the_same_court() -> None:
    target = _transform(center=(64.0, 64.0), rotation=0.2)
    half_turn = torch.tensor(CAMERA_VIEW_HALF_TURN_INDEX)
    prediction = target[half_turn]
    predictions = _instances([prediction], [(64.0, 64.0)])
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["instance_tp"] == 1.0
    assert result["instance_kp_mean_error_px"] == pytest.approx(0.0, abs=1.0e-6)
    assert result["instance_kp_pck_2px"] == 1.0
    assert result["instance_kp_pck_4px"] == 1.0
    assert result["sim2_rotation_error_deg"] == pytest.approx(0.0, abs=1.0e-5)


def test_false_positive_and_missed_court_are_counted_explicitly() -> None:
    matched = _transform(center=(40.0, 40.0))
    missed = _transform(center=(80.0, 80.0))
    false_positive = _transform(center=(115.0, 15.0))
    predictions = _instances(
        [matched, false_positive], [(40.0, 40.0), (115.0, 15.0)]
    )
    keypoints, visibility, centers, counts = _targets(
        [matched, missed], [(40.0, 40.0), (80.0, 80.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["instance_tp"] == 1.0
    assert result["instance_fp"] == 1.0
    assert result["instance_fn"] == 1.0
    assert result["false_positive_count"] == 1.0
    assert result["instance_precision"] == 0.5
    assert result["instance_recall"] == 0.5
    assert result["instance_f1"] == 0.5


def test_both_empty_has_explicit_finite_conventions() -> None:
    predictions = _instances([], [])
    keypoints, visibility, centers, counts = _targets([], [])

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["instance_count_accuracy"] == 1.0
    assert result["instance_precision"] == 0.0
    assert result["instance_recall"] == 0.0
    assert result["instance_f1"] == 0.0
    assert result["instance_kp_pck_2px"] == 0.0
    assert all(math.isfinite(value) for value in result.values())


@pytest.mark.parametrize("prediction_only", [False, True])
def test_one_sided_empty_uses_finite_pessimistic_errors(
    prediction_only: bool,
) -> None:
    court = _transform(center=(64.0, 64.0))
    predictions = (
        _instances([court], [(64.0, 64.0)])
        if prediction_only
        else _instances([], [])
    )
    keypoints, visibility, centers, counts = (
        _targets([], [])
        if prediction_only
        else _targets([court], [(64.0, 64.0)])
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    diagonal = math.hypot(127.0, 127.0)
    assert result["matched_instance_count"] == 0.0
    assert result["matched_center_mean_error_px"] == pytest.approx(diagonal)
    assert result["instance_kp_mean_error_px"] == pytest.approx(diagonal)
    assert result["instance_kp_pck_4px"] == 0.0
    assert result["sim2_rotation_error_deg"] == 180.0
    assert result["sim2_scale_relative_error"] == 1.0
    if prediction_only:
        assert result["instance_fp"] == 1.0
        assert result["instance_fn"] == 0.0
    else:
        assert result["instance_fp"] == 0.0
        assert result["instance_fn"] == 1.0


def test_sim2_residual_metrics_recover_transform_differences() -> None:
    target = _transform(center=(64.0, 64.0), rotation=0.1, scale=2.0)
    prediction = _transform(center=(67.0, 60.0), rotation=0.2, scale=2.2)
    predictions = _instances([prediction], [(67.0, 60.0)])
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
        match_max_error_px=20.0,
    )

    assert result["sim2_pair_count"] == 1.0
    assert result["sim2_translation_error_px"] == pytest.approx(5.0, abs=1.0e-4)
    assert result["sim2_rotation_error_deg"] == pytest.approx(
        math.degrees(0.1), abs=1.0e-4
    )
    assert result["sim2_scale_relative_error"] == pytest.approx(0.1, abs=1.0e-5)


def test_two_of_fourteen_exact_keypoints_cannot_score_as_a_detected_court() -> None:
    target = _transform(center=(64.0, 64.0))
    valid = torch.zeros((1, 14), dtype=torch.bool)
    valid[:, :2] = True
    predictions = CourtInstances(
        (
            CourtInstanceBatch(
                keypoints_px=target.unsqueeze(0),
                scores=torch.ones((1, 14)),
                valid=valid,
                centers_px=torch.tensor([[64.0, 64.0]]),
            ),
        )
    )
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    diagonal = math.hypot(127.0, 127.0)
    assert result["instance_tp"] == 0.0
    assert result["instance_fp"] == 1.0
    assert result["instance_fn"] == 1.0
    assert result["instance_f1"] == 0.0
    assert result["instance_kp_pck_2px"] == 0.0
    assert result["visible_kp_gt"] == 14.0
    assert result["visible_kp_matched"] == 0.0
    assert result["visible_kp_coverage"] == 0.0
    assert result["coverage_gate_pass_rate"] == 0.0
    assert result["insufficient_coverage_count"] == 1.0
    assert result["sim2_pair_count"] == 0.0
    assert result["sim2_unavailable_count"] == 1.0
    assert result["sim2_translation_error_px"] == pytest.approx(diagonal)
    assert result["sim2_rotation_error_deg"] == 180.0
    assert result["sim2_scale_relative_error"] == 1.0


def test_missing_visible_keypoints_remain_in_pck_denominator() -> None:
    target = _transform(center=(64.0, 64.0))
    valid = torch.zeros((1, 14), dtype=torch.bool)
    valid[:, :7] = True
    predictions = CourtInstances(
        (
            CourtInstanceBatch(
                keypoints_px=target.unsqueeze(0),
                scores=torch.ones((1, 14)),
                valid=valid,
                centers_px=torch.tensor([[64.0, 64.0]]),
            ),
        )
    )
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["instance_tp"] == 1.0
    assert result["visible_kp_gt"] == 14.0
    assert result["visible_kp_matched"] == 7.0
    assert result["visible_kp_coverage"] == 0.5
    assert result["instance_kp_pck_2px"] == 0.5
    assert result["instance_kp_pck_4px"] == 0.5


def test_sim2_unavailable_pair_contributes_explicit_penalty_to_mean() -> None:
    target = _transform(center=(64.0, 64.0))
    valid = torch.zeros((1, 14), dtype=torch.bool)
    valid[:, :7] = True
    predictions = CourtInstances(
        (
            CourtInstanceBatch(
                keypoints_px=target.unsqueeze(0),
                scores=torch.ones((1, 14)),
                valid=valid,
                centers_px=torch.tensor([[64.0, 64.0]]),
            ),
        )
    )
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
        minimum_sim2_keypoints=8,
    )

    assert result["instance_tp"] == 1.0
    assert result["sim2_pair_count"] == 0.0
    assert result["sim2_evaluation_count"] == 1.0
    assert result["sim2_unavailable_count"] == 1.0
    assert result["sim2_translation_error_px"] == pytest.approx(
        math.hypot(127.0, 127.0)
    )
    assert result["sim2_rotation_error_deg"] == 180.0
    assert result["sim2_scale_relative_error"] == 1.0


def test_sim2_penalty_is_not_dropped_when_another_pair_is_available() -> None:
    first = _transform(center=(40.0, 40.0))
    second = _transform(center=(88.0, 88.0))
    valid = torch.ones((2, 14), dtype=torch.bool)
    valid[1, 7:] = False
    predictions = CourtInstances(
        (
            CourtInstanceBatch(
                keypoints_px=torch.stack((first, second)),
                scores=torch.ones((2, 14)),
                valid=valid,
                centers_px=torch.tensor([[40.0, 40.0], [88.0, 88.0]]),
            ),
        )
    )
    keypoints, visibility, centers, counts = _targets(
        [first, second], [(40.0, 40.0), (88.0, 88.0)]
    )

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
        minimum_sim2_keypoints=8,
    )

    assert result["instance_tp"] == 2.0
    assert result["sim2_pair_count"] == 1.0
    assert result["sim2_evaluation_count"] == 2.0
    assert result["sim2_unavailable_count"] == 1.0
    assert result["sim2_translation_error_px"] == pytest.approx(
        math.hypot(127.0, 127.0) / 2.0
    )
    assert result["sim2_rotation_error_deg"] == 90.0
    assert result["sim2_scale_relative_error"] == 0.5


def test_target_with_too_few_visible_keypoints_cannot_pass_coverage_gate() -> None:
    target = _transform(center=(64.0, 64.0))
    predictions = _instances([target], [(64.0, 64.0)])
    keypoints, visibility, centers, counts = _targets(
        [target], [(64.0, 64.0)]
    )
    visibility[:, :, 3:] = False

    result = instance_alignment_metrics(
        predictions,
        keypoints,
        visibility,
        centers=centers,
        num_courts=counts,
        image_size=(128, 128),
    )

    assert result["visible_kp_gt"] == 3.0
    assert result["instance_tp"] == 0.0
    assert result["instance_fp"] == 1.0
    assert result["instance_fn"] == 1.0
    assert result["insufficient_coverage_count"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_common_keypoints": True},
        {"minimum_common_keypoints": 0},
        {"minimum_visible_keypoints": 15},
        {"minimum_visible_fraction": 1},
        {"minimum_visible_fraction": 0.0},
        {"minimum_sim2_keypoints": 1},
    ],
)
def test_metric_coverage_options_are_strict(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CourtAlignmentMetrics(**kwargs)  # type: ignore[arg-type]


def test_metric_coverage_defaults_are_explicit() -> None:
    metrics = CourtAlignmentMetrics()

    assert metrics.minimum_common_keypoints == 4
    assert metrics.minimum_visible_keypoints == 4
    assert metrics.minimum_visible_fraction == 0.5
    assert metrics.minimum_sim2_keypoints == 4
