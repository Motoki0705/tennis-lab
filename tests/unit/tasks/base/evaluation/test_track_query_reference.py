"""Reference-frame training metric tests."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.base.evaluation.track_query_reference import (
    AxisWisePositionError,
    PairedReferenceEvaluationError,
    compute_axis_wise_position_error,
    compute_heading_error_radians,
    compute_paired_reference_position_metrics,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)


def test_y_sign_axis_error_and_local_index_metrics_are_exact() -> None:
    prediction = torch.tensor(
        [
            [[2.0, 3.0, 1.0], [0.0, -1.0, 2.0]],
            [[1.0, -4.0, 3.0], [4.0, 2.0, 0.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 2.0, 1.0], [0.0, -2.0, 4.0]],
            [[3.0, 4.0, 1.0], [2.0, 1.0, 0.0]],
        ]
    )
    reference_index = torch.tensor([2, 0], dtype=torch.int64)

    assert compute_y_sign_accuracy(prediction, target) == pytest.approx(0.75)
    assert compute_axis_wise_position_error(prediction, target) == (
        AxisWisePositionError(x=1.25, y=2.75, z=1.0)
    )
    report = compute_paired_reference_position_metrics(
        prediction,
        target,
        reference_index,
    )
    expected_sample_0 = (
        torch.linalg.vector_norm(prediction[0] - target[0], dim=-1).mean().item()
    )
    expected_sample_1 = (
        torch.linalg.vector_norm(prediction[1] - target[1], dim=-1).mean().item()
    )
    assert report.local_reference_index_error == pytest.approx(
        {0: expected_sample_1, 2: expected_sample_0}
    )


def test_amp_mixed_position_heading_and_strata_use_stable_float32() -> None:
    target_position = torch.tensor(
        [[[1.0, 2.0, 1.0]], [[-1.0, -2.0, 1.0]]],
        dtype=torch.float32,
    )
    prediction_position = torch.tensor(
        [[[2.0, 3.0, 1.0]], [[0.0, -1.0, 1.0]]],
        dtype=torch.bfloat16,
    )
    reference_index = torch.tensor([0, 1], dtype=torch.int64)
    prediction_heading = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.bfloat16,
    )
    target_heading = prediction_heading.to(torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y_sign = compute_y_sign_accuracy(prediction_position, target_position)
        axis_error = compute_axis_wise_position_error(
            prediction_position,
            target_position,
        )
        report = compute_paired_reference_position_metrics(
            prediction_position,
            target_position,
            reference_index,
        )
        heading_error = compute_heading_error_radians(
            prediction_heading,
            target_heading,
        )
        stratified = stratify_metric_by_reference_view_index(
            torch.tensor([1.0, 1.0078125], dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.int64),
        )

    assert y_sign == pytest.approx(1.0)
    assert axis_error == AxisWisePositionError(x=1.0, y=1.0, z=0.0)
    assert report.local_reference_index_error == pytest.approx(
        {0: math.sqrt(2.0), 1: math.sqrt(2.0)}
    )
    assert heading_error == pytest.approx(0.0)
    assert stratified == pytest.approx({0: 1.00390625})


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (torch.zeros(1, 2, 3), "shapes differ"),
        (torch.zeros(1, 1, 3, dtype=torch.int64), "floating dtypes"),
        (torch.empty(1, 1, 3, device="meta"), "share a device"),
    ],
)
def test_mixed_dtype_pair_contract_still_rejects_semantic_mismatches(
    target: torch.Tensor,
    message: str,
) -> None:
    prediction = torch.zeros(1, 1, 3, dtype=torch.bfloat16)

    with pytest.raises(PairedReferenceEvaluationError, match=message):
        compute_axis_wise_position_error(prediction, target)


def test_y_sign_uses_explicit_mask_and_rejects_only_mid_plane_targets() -> None:
    prediction = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    assert compute_y_sign_accuracy(
        prediction,
        target,
        zero_tolerance=1e-9,
    ) == pytest.approx(1.0)
    with pytest.raises(PairedReferenceEvaluationError, match="no targets"):
        compute_y_sign_accuracy(
            prediction[:, :1],
            target[:, :1],
            zero_tolerance=1e-9,
        )


def test_heading_error_is_radians_and_rejects_zero_vectors() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    assert compute_heading_error_radians(prediction, target) == pytest.approx(
        math.pi / 4,
        abs=1e-12,
    )
    with pytest.raises(PairedReferenceEvaluationError, match="non-zero"):
        compute_heading_error_radians(torch.zeros_like(prediction), target)


def test_local_index_stratification_rejects_padding_and_nonfinite_values() -> None:
    values = torch.tensor([1.0, 3.0, 5.0])
    indices = torch.tensor([1, 0, 1], dtype=torch.int64)
    assert stratify_metric_by_reference_view_index(values, indices) == {
        0: 3.0,
        1: 3.0,
    }
    with pytest.raises(PairedReferenceEvaluationError, match="negative"):
        stratify_metric_by_reference_view_index(
            values,
            torch.tensor([1, -1, 1]),
        )
    with pytest.raises(PairedReferenceEvaluationError, match="finite"):
        stratify_metric_by_reference_view_index(
            torch.tensor([1.0, float("nan"), 2.0]),
            indices,
        )
