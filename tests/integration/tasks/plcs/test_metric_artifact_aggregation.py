"""Integration coverage for standard PLCS diagnostic artifact aggregation."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.metric_logging import WeightedMetricAccumulator
from src.tasks.plcs.training.lightning_module import PLCS_TRAJECTORY_METRIC_CONTRACT
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.utils.schema.court_normalization import normalize_court_position


class _ArtifactModule(ManualGANSupportMixin):
    metric_logging_contract = PLCS_TRAJECTORY_METRIC_CONTRACT

    def __init__(self, tracker: PLCSMetrics) -> None:
        self.tracker = tracker
        self._test_metric_diagnostic_accumulator = WeightedMetricAccumulator()
        self.saved: dict[str, Any] = {}

    def _metric_tracker_for_stage(self, stage: str) -> PLCSMetrics:
        assert stage == "test"
        return self.tracker

    def save_test_predictions(self, **kwargs: Any) -> None:
        self.saved = kwargs

    def log(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _diagnostic_artifact(
    partitions: tuple[tuple[int, ...], ...],
) -> dict[str, float]:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )
    module = _ArtifactModule(tracker)
    target_m = torch.zeros(4, 2, 3)
    target_m[..., 1] = 2.0
    prediction_m = target_m.clone()
    prediction_m[..., 0] = torch.tensor(
        [[1.0, 1.0], [10.0, 10.0], [3.0, 100.0], [5.0, 5.0]]
    )
    rotation = torch.zeros(4, 2, 2)
    rotation[..., 0] = 1.0
    padding_mask = torch.tensor(
        [[False, False], [False, False], [False, True], [False, False]]
    )
    reference_view_index = torch.tensor([0, 1, 0, 0], dtype=torch.int64)

    normalized_prediction = normalize_court_position(prediction_m)
    normalized_target = normalize_court_position(target_m)
    for partition in partitions:
        indices = torch.tensor(partition, dtype=torch.int64)
        batch_metrics = tracker.update(
            normalized_prediction[indices],
            rotation[indices],
            normalized_target[indices],
            rotation[indices],
            padding_mask=padding_mask[indices],
            reference_view_index=reference_view_index[indices],
        )
        module._test_metric_diagnostic_accumulator.update(
            batch_metrics,
            weight=len(partition),
        )

    module.on_test_epoch_end()
    return cast("dict[str, float]", module.saved["diagnostic_metrics"])


def _y_sign_diagnostic_artifact(
    partitions: tuple[tuple[int, ...], ...],
) -> dict[str, float]:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )
    module = _ArtifactModule(tracker)
    target_m = torch.zeros(2, 13, 3)
    prediction_m = torch.zeros_like(target_m)

    # Sample 0: ten eligible correct frames, one valid mid-plane frame, and
    # one padded eligible frame whose incorrect sign must not count.
    target_m[0, :10, 1] = 1.0
    prediction_m[0, :10, 1] = 1.0
    prediction_m[0, 10, 1] = -1.0
    target_m[0, 11, 1] = 1.0
    prediction_m[0, 11, 1] = -1.0

    # Sample 1: one eligible incorrect frame and one valid mid-plane frame.
    target_m[1, 0, 1] = 1.0
    prediction_m[1, 0, 1] = -1.0
    prediction_m[1, 1, 1] = 1.0

    padding_mask = torch.ones(2, 13, dtype=torch.bool)
    padding_mask[0, :11] = False
    padding_mask[1, :2] = False
    rotation = torch.zeros(2, 13, 2)
    rotation[..., 0] = 1.0
    reference_view_index = torch.tensor([0, 1], dtype=torch.int64)
    normalized_prediction = normalize_court_position(prediction_m)
    normalized_target = normalize_court_position(target_m)

    for partition in partitions:
        indices = torch.tensor(partition, dtype=torch.int64)
        batch_metrics = tracker.update(
            normalized_prediction[indices],
            rotation[indices],
            normalized_target[indices],
            rotation[indices],
            padding_mask=padding_mask[indices],
            reference_view_index=reference_view_index[indices],
        )
        module._test_metric_diagnostic_accumulator.update(
            batch_metrics,
            weight=len(partition),
        )

    module.on_test_epoch_end()
    return cast("dict[str, float]", module.saved["diagnostic_metrics"])


def _optional_y_sign_artifact(
    partitions: tuple[tuple[int, ...], ...],
    *,
    all_mid_plane: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    tracker = PLCSMetrics(
        position_threshold_m=0.5,
        angle_threshold_deg=15.0,
    )
    module = _ArtifactModule(tracker)
    target_m = torch.zeros(2, 3, 3)
    prediction_m = torch.zeros_like(target_m)
    prediction_m[..., 0] = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 100.0]])
    prediction_m[:, :2, 1] = torch.tensor([[0.5, -0.5], [0.25, -0.25]])
    if not all_mid_plane:
        target_m[0, 0, 1] = 1.0
        prediction_m[0, 0, 1] = 1.0
    # This padded sign mismatch must never make Y-sign evidence eligible.
    target_m[0, 2, 1] = 1.0
    prediction_m[0, 2, 1] = -1.0

    padding_mask = torch.tensor(
        [[False, False, True], [False, False, True]],
        dtype=torch.bool,
    )
    rotation = torch.zeros(2, 3, 2)
    rotation[..., 0] = 1.0
    reference_view_index = torch.tensor([0, 1], dtype=torch.int64)
    normalized_prediction = normalize_court_position(prediction_m)
    normalized_target = normalize_court_position(target_m)

    for partition in partitions:
        indices = torch.tensor(partition, dtype=torch.int64)
        batch_metrics = tracker.update(
            normalized_prediction[indices],
            rotation[indices],
            normalized_target[indices],
            rotation[indices],
            padding_mask=padding_mask[indices],
            reference_view_index=reference_view_index[indices],
        )
        module._test_metric_diagnostic_accumulator.update(
            batch_metrics,
            weight=len(partition),
        )

    module.on_test_epoch_end()
    return (
        cast("dict[str, float]", module.saved["metrics"]),
        cast("dict[str, float]", module.saved["diagnostic_metrics"]),
    )


def test_reference_strata_artifact_is_invariant_to_batch_partition() -> None:
    batch_size_two = _diagnostic_artifact(((0, 1), (2, 3)))
    batch_size_one = _diagnostic_artifact(((0,), (1,), (2,), (3,)))

    assert batch_size_two == pytest.approx(batch_size_one)
    assert batch_size_two["reference_index_0_position_error_m"] == pytest.approx(3.0)
    assert batch_size_two["reference_index_1_position_error_m"] == pytest.approx(
        10.0
    )
    assert "heading_error_deg" not in batch_size_two


def test_y_sign_artifact_uses_eligible_frame_sufficient_statistics() -> None:
    batch_size_two = _y_sign_diagnostic_artifact(((0, 1),))
    batch_size_one = _y_sign_diagnostic_artifact(((0,), (1,)))

    assert batch_size_two == pytest.approx(batch_size_one)
    assert batch_size_two["y_sign_accuracy"] == pytest.approx(10.0 / 11.0)


def test_optional_y_sign_evidence_is_partition_invariant() -> None:
    combined_metrics, combined_diagnostics = _optional_y_sign_artifact(
        ((0, 1),),
        all_mid_plane=False,
    )
    split_metrics, split_diagnostics = _optional_y_sign_artifact(
        ((0,), (1,)),
        all_mid_plane=False,
    )

    assert combined_metrics == pytest.approx(split_metrics)
    assert combined_diagnostics == pytest.approx(split_diagnostics)
    assert combined_diagnostics["y_sign_accuracy"] == pytest.approx(1.0)
    assert "reference_index_1_position_error_m" in combined_diagnostics


def test_all_mid_plane_epoch_omits_only_undefined_y_sign_evidence() -> None:
    combined_metrics, combined_diagnostics = _optional_y_sign_artifact(
        ((0, 1),),
        all_mid_plane=True,
    )
    split_metrics, split_diagnostics = _optional_y_sign_artifact(
        ((0,), (1,)),
        all_mid_plane=True,
    )

    assert combined_metrics == pytest.approx(split_metrics)
    assert combined_diagnostics == pytest.approx(split_diagnostics)
    assert "y_sign_accuracy" not in combined_diagnostics
    assert set(combined_metrics) == {
        "position_error_m",
        "angular_error_deg",
        "position_accuracy_0.5m",
        "angle_accuracy_15deg",
    }
    assert "x_error_m" in combined_diagnostics
    assert "reference_index_0_position_error_m" in combined_diagnostics
    assert "reference_index_1_position_error_m" in combined_diagnostics
