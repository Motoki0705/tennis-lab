"""Reference-conditioned PLCS tracking metric tests."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.training.metric_logging import (
    MetricStatisticsAccumulator,
    compute_scalar_metric_statistics,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.training.tracking_metrics import (
    plcs_tracking_metrics,
    plcs_tracking_statistics,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _tracking_config() -> TrackingMetricConfig:
    return TrackingMetricConfig(
        presence_threshold=0.5,
        duplicate_distance=0.05,
        id_switch_distance=0.05,
    )


def _tracking_batch(
    errors_x_m: tuple[tuple[float, ...], ...],
    *,
    target_y_m: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build one-view clips with independent valid lengths for aggregation tests."""
    batch_size = len(errors_x_m)
    frames = max(len(errors) for errors in errors_x_m)
    scale = torch.tensor(COURT_COORD_SCALE_XYZ)
    target_position = torch.zeros(batch_size, frames, 1, 3)
    target_position[:, :, 0, 1] = target_y_m / scale[1]
    prediction_position = torch.zeros_like(target_position)
    padding_mask = torch.ones(batch_size, 1, frames, dtype=torch.bool)
    target_presence = torch.zeros(batch_size, frames, 1, dtype=torch.bool)
    prediction_rotation = torch.zeros(batch_size, frames, 1, 2)
    target_rotation = torch.zeros_like(prediction_rotation)
    target_rotation[..., 0] = 1.0
    for sample_index, errors in enumerate(errors_x_m):
        valid_length = len(errors)
        padding_mask[sample_index, 0, :valid_length] = False
        target_presence[sample_index, :valid_length, 0] = True
        prediction_position[sample_index, :valid_length, 0, 1] = (
            target_y_m / scale[1]
        )
        prediction_position[sample_index, :valid_length, 0, 0] = (
            torch.tensor(errors) / scale[0]
        )
        prediction_rotation[sample_index, :valid_length, 0, 0] = 1.0
    prediction = {
        "position": prediction_position,
        "rotation": prediction_rotation,
        "presence_logits": torch.full((batch_size, frames, 1), 20.0),
    }
    batch = {
        "target_position": target_position,
        "target_rotation": target_rotation,
        "target_presence": target_presence,
        "target_instance_id": torch.zeros(
            batch_size, frames, 1, dtype=torch.int64
        ),
        "padding_mask": padding_mask,
    }
    assignments = [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )
        for _ in range(batch_size)
    ]
    return prediction, batch, assignments


def test_tracking_statistics_use_matched_frame_denominators() -> None:
    prediction, batch, assignments = _tracking_batch(((1.0, 3.0, 5.0), (7.0,)))
    prediction["rotation"][0, 1, 0] = torch.tensor([0.0, 1.0])
    prediction["rotation"][1, 0, 0] = torch.tensor([0.0, 1.0])

    statistics = plcs_tracking_statistics(
        prediction,
        batch,
        assignments,
        config=_tracking_config(),
    )

    assert statistics["position_error_m"].numerator.item() == pytest.approx(16.0)
    assert statistics["position_error_m"].denominator.item() == pytest.approx(4.0)
    assert statistics["angular_error_deg"].numerator.item() == pytest.approx(180.0)
    assert statistics["angular_error_deg"].denominator.item() == pytest.approx(4.0)
    metrics = plcs_tracking_metrics(
        prediction,
        batch,
        assignments,
        config=_tracking_config(),
    )
    assert metrics["position_error_m"].item() == pytest.approx(4.0)
    assert metrics["angular_error_deg"].item() == pytest.approx(45.0)


def test_position_and_angular_metrics_match_b2_and_b1_plus_b1() -> None:
    prediction, batch, assignments = _tracking_batch(((1.0, 3.0, 5.0), (7.0,)))
    prediction["rotation"][0, 1, 0] = torch.tensor([0.0, 1.0])
    prediction["rotation"][1, 0, 0] = torch.tensor([0.0, 1.0])
    combined = MetricStatisticsAccumulator()
    combined.update(
        plcs_tracking_statistics(
            prediction,
            batch,
            assignments,
            config=_tracking_config(),
            reference_view_index=torch.tensor([0, 0], dtype=torch.int64),
        )
    )

    first_prediction, first_batch, first_assignments = _tracking_batch(
        ((1.0, 3.0, 5.0),)
    )
    first_prediction["rotation"][0, 1, 0] = torch.tensor([0.0, 1.0])
    second_prediction, second_batch, second_assignments = _tracking_batch(((7.0,),))
    second_prediction["rotation"][0, 0, 0] = torch.tensor([0.0, 1.0])
    split = MetricStatisticsAccumulator()
    for sample_prediction, sample_batch, sample_assignments in (
        (first_prediction, first_batch, first_assignments),
        (second_prediction, second_batch, second_assignments),
    ):
        split.update(
            plcs_tracking_statistics(
                sample_prediction,
                sample_batch,
                sample_assignments,
                config=_tracking_config(),
                reference_view_index=torch.tensor([0], dtype=torch.int64),
            )
        )

    assert split.compute() == pytest.approx(combined.compute())
    assert combined.compute()["position_error_m"] == pytest.approx(4.0)
    assert combined.compute()["angular_error_deg"] == pytest.approx(45.0)
    assert combined.compute()["reference_index_0_position_error_m"] == (
        pytest.approx(5.0)
    )


def test_tracking_statistics_preserve_reference_strata_as_sample_means() -> None:
    prediction, batch, assignments = _tracking_batch(((1.0, 3.0, 5.0), (7.0,)))
    reference_index = torch.tensor([0, 0], dtype=torch.int64)

    statistics = plcs_tracking_statistics(
        prediction,
        batch,
        assignments,
        config=_tracking_config(),
        reference_view_index=reference_index,
    )

    stratum = statistics["reference_index_0_position_error_m"]
    assert stratum.numerator.item() == pytest.approx(10.0)
    assert stratum.denominator.item() == pytest.approx(2.0)
    assert statistics["x_error_m"].numerator.item() == pytest.approx(16.0)
    assert statistics["x_error_m"].denominator.item() == pytest.approx(4.0)
    assert statistics["y_sign_accuracy"].numerator.item() == pytest.approx(4.0)
    assert statistics["y_sign_accuracy"].denominator.item() == pytest.approx(4.0)
    assert compute_scalar_metric_statistics(
        statistics,
        zero_denominator_value=0.0,
    )["reference_index_0_position_error_m"].item() == pytest.approx(5.0)


def test_tracking_statistics_have_zero_denominators_for_all_padding() -> None:
    # ``_tracking_batch`` needs one frame to construct an all-padding clip.
    prediction, batch, assignments = _tracking_batch(((0.0,),))
    batch["padding_mask"][:] = True
    batch["target_presence"][:] = False

    statistics = plcs_tracking_statistics(
        prediction,
        batch,
        assignments,
        config=_tracking_config(),
        reference_view_index=torch.tensor([0], dtype=torch.int64),
    )
    metrics = plcs_tracking_metrics(
        prediction,
        batch,
        assignments,
        config=_tracking_config(),
        reference_view_index=torch.tensor([0], dtype=torch.int64),
    )

    for key in (
        "position_error_m",
        "angular_error_deg",
        "x_error_m",
        "y_sign_accuracy",
    ):
        assert statistics[key].denominator.item() == pytest.approx(0.0)
        assert metrics[key].item() == pytest.approx(0.0)
    assert "heading_error_deg" not in statistics
    assert "heading_error_deg" not in metrics


def test_all_padding_sample_does_not_dilute_a_mixed_plcs_batch() -> None:
    valid_prediction, valid_batch, valid_assignments = _tracking_batch(((1.0, 3.0),))
    valid = MetricStatisticsAccumulator()
    valid.update(
        plcs_tracking_statistics(
            valid_prediction,
            valid_batch,
            valid_assignments,
            config=_tracking_config(),
            reference_view_index=torch.tensor([0], dtype=torch.int64),
        )
    )
    mixed_prediction, mixed_batch, mixed_assignments = _tracking_batch(
        ((1.0, 3.0), (100.0, 200.0))
    )
    mixed_batch["padding_mask"][1] = True
    mixed_batch["target_presence"][1] = False
    mixed = MetricStatisticsAccumulator()
    mixed.update(
        plcs_tracking_statistics(
            mixed_prediction,
            mixed_batch,
            mixed_assignments,
            config=_tracking_config(),
            reference_view_index=torch.tensor([0, 1], dtype=torch.int64),
        )
    )

    assert mixed.compute() == pytest.approx(valid.compute())
    assert "reference_index_1_position_error_m" not in mixed.compute()


def test_tracking_metrics_report_y_sign_axes_and_local_index_without_heading_alias() -> (
    None
):
    target_position = torch.tensor(
        [[[[0.2, 0.3, 0.1]]], [[[-0.2, -0.3, 0.1]]]]
    )
    prediction = {
        "position": target_position + torch.tensor([0.1, -0.1, 0.2]),
        "rotation": torch.tensor(
            [[[[0.0, 1.0]]], [[[1.0, 0.0]]]],
        ),
        "presence_logits": torch.full((2, 1, 1), 10.0),
    }
    batch = {
        "target_position": target_position,
        "target_rotation": torch.tensor(
            [[[[1.0, 0.0]]], [[[1.0, 0.0]]]],
        ),
        "target_presence": torch.ones(2, 1, 1, dtype=torch.bool),
        "target_instance_id": torch.tensor([[[10]], [[20]]], dtype=torch.int64),
        "padding_mask": torch.zeros(2, 1, 1, dtype=torch.bool),
    }
    assignments = [
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
    ]

    metrics = plcs_tracking_metrics(
        prediction,
        batch,
        assignments,
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.05,
            id_switch_distance=0.05,
        ),
        reference_view_index=torch.tensor([0, 1], dtype=torch.int64),
    )

    assert metrics["y_sign_accuracy"].item() == pytest.approx(1.0)
    assert metrics["angular_error_deg"].item() == pytest.approx(45.0)
    assert "heading_error_deg" not in metrics
    assert metrics["reference_index_0_position_error_m"].item() > 0.0
    assert metrics["reference_index_1_position_error_m"].item() > 0.0
    assert metrics["x_error_m"].item() > 0.0
    assert metrics["y_error_m"].item() > 0.0
    assert metrics["z_error_m"].item() > 0.0
