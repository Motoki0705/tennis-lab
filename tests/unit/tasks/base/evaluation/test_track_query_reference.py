"""Paired-reference evaluation metric tests."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.base.evaluation.track_query_reference import (
    AxisWisePositionError,
    PairedReferenceEvaluationError,
    PairedReferenceKey,
    ReferenceTransformQuantity,
    compute_axis_wise_position_error,
    compute_heading_error_radians,
    compute_paired_reference_position_metrics,
    compute_reference_transform_consistency_error,
    compute_y_sign_accuracy,
    stratify_metric_by_reference_view_index,
)
from src.tasks.base.generate_dataset.court_view import (
    CAMERA_VIEW_V2_SELECTOR,
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_reference_frame_provenance,
    court_headings_physical_to_target,
    court_points_physical_to_target,
    court_vectors_physical_to_target,
    court_world_joints_physical_to_target,
    resolve_court_keypoint_contract,
)


def _paired_provenance() -> tuple[
    CourtReferenceFrameProvenance,
    CourtReferenceFrameProvenance,
]:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    views = (
        build_court_view_record(
            camera_id="negative",
            camera_center_court_m=(0.0, -5.0, 2.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="positive",
            camera_center_court_m=(0.0, 5.0, 2.0),
            contract=contract,
        ),
    )
    return (
        build_reference_frame_provenance(
            views,
            reference_camera_id="negative",
        ),
        build_reference_frame_provenance(
            views,
            reference_camera_id="positive",
        ),
    )


def test_paired_key_requires_same_complete_view_set_and_local_ordering() -> None:
    key = PairedReferenceKey(
        scene_id="scene_a",
        view_camera_ids=("a", "b"),
        local_ordering=("b", "a"),
    )
    assert key.local_ordering == ("b", "a")
    with pytest.raises(PairedReferenceEvaluationError, match="permutation"):
        PairedReferenceKey(
            scene_id="scene_a",
            view_camera_ids=("a", "b"),
            local_ordering=("a", "c"),
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


@pytest.mark.parametrize(
    ("quantity", "physical"),
    [
        ("point", torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)),
        ("vector", torch.tensor([[0.5, -2.0, 1.0]], dtype=torch.float64)),
        ("heading", torch.tensor([[0.0, 1.0]], dtype=torch.float64)),
        (
            "world_joints",
            torch.tensor([[[-1.0, 2.0, 0.5], [2.0, -3.0, 1.0]]], dtype=torch.float64),
        ),
    ],
)
def test_opposite_reference_transform_consistency_uses_geometry_authority(
    quantity: ReferenceTransformQuantity,
    physical: torch.Tensor,
) -> None:
    negative, positive = _paired_provenance()
    if quantity == "point":
        first = court_points_physical_to_target(physical, negative)
        second = court_points_physical_to_target(physical, positive)
    elif quantity == "vector":
        first = court_vectors_physical_to_target(physical, negative)
        second = court_vectors_physical_to_target(physical, positive)
    elif quantity == "heading":
        first = court_headings_physical_to_target(physical, negative)
        second = court_headings_physical_to_target(physical, positive)
    else:
        first = court_world_joints_physical_to_target(physical, negative)
        second = court_world_joints_physical_to_target(physical, positive)
    assert isinstance(first, torch.Tensor)
    assert isinstance(second, torch.Tensor)

    error = compute_reference_transform_consistency_error(
        first,
        negative,
        second,
        positive,
        quantity=quantity,
    )
    assert error == pytest.approx(0.0, abs=1e-12)


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
