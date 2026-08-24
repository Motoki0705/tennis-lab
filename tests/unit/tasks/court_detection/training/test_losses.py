"""Pure tensor tests for Court query losses."""

from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from src.tasks.court_detection.geometry.pose import (
    canonical_semantic_court_points_batched,
    decode_pose10d_strict,
    project_predicted_canonical_points,
)
from src.tasks.court_detection.training.losses import (
    CourtConsistencyGradientFlow,
    CourtKeypointPoseConsistencyLoss,
    consistency_effective_weight,
    query_keypoint_pose_consistency_loss,
)
from src.utils.data.heatmaps import heatmaps_to_soft_argmax


def _pad_to_kp14(
    dense: torch.Tensor,
    pose: torch.Tensor,
    depth: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    padding = 14 - dense.shape[1]
    if padding < 0:
        raise ValueError("Test input cannot exceed KP14.")
    return (
        torch.cat((dense, dense.new_zeros((dense.shape[0], padding, 2))), dim=1),
        torch.cat((pose, pose.new_zeros((pose.shape[0], padding, 2))), dim=1),
        torch.cat((depth, depth.new_ones((depth.shape[0], padding))), dim=1),
        torch.cat(
            (
                visible,
                torch.zeros(
                    (visible.shape[0], padding),
                    dtype=torch.bool,
                    device=visible.device,
                ),
            ),
            dim=1,
        ),
    )


def _consistency_loss(
    dense: torch.Tensor,
    pose: torch.Tensor,
    depth: torch.Tensor,
    visible: torch.Tensor,
    *,
    image_size: torch.Tensor | None = None,
    cheirality_weight: float = 0.0,
    gradient_flow: CourtConsistencyGradientFlow = "both",
) -> CourtKeypointPoseConsistencyLoss:
    dense, pose, depth, visible = _pad_to_kp14(dense, pose, depth, visible)
    return query_keypoint_pose_consistency_loss(
        dense,
        pose,
        depth,
        torch.tensor([[3, 4]], dtype=torch.long)
        if image_size is None
        else image_size,
        visible,
        huber_delta=0.1,
        min_depth_m=0.1,
        depth_scale_m=0.5,
        cheirality_weight=cheirality_weight,
        gradient_flow=gradient_flow,
    )


def test_coordinate_loss_is_diagonal_normalized_huber_distance() -> None:
    dense = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
    pose = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    depth = torch.tensor([[10.0]], dtype=torch.float64)
    result = _consistency_loss(
        dense,
        pose,
        depth,
        torch.tensor([[True]]),
        cheirality_weight=0.0,
    )

    # diagonal=5, residual=1 => r=.2 and Huber(.2; delta=.1)=.015
    assert float(result.coordinate) == pytest.approx(0.015)
    assert float(result.auxiliary) == pytest.approx(0.015)
    assert float(result.mean_distance_px) == pytest.approx(1.0)


def test_fixed_gt_visibility_controls_both_reductions() -> None:
    dense = torch.tensor(
        [[[0.0, 0.0], [1000.0, 1000.0]], [[5.0, 5.0], [6.0, 6.0]]]
    )
    pose = torch.tensor(
        [[[1.0, 0.0], [-1000.0, -1000.0]], [[-5.0, -5.0], [-6.0, -6.0]]]
    )
    depth = torch.tensor([[1.0, -1000.0], [-1000.0, -1000.0]])
    visible = torch.tensor([[True, False], [False, False]])
    image_size = torch.tensor([[3, 4], [300, 400]], dtype=torch.long)

    result = _consistency_loss(
        dense,
        pose,
        depth,
        visible,
        image_size=image_size,
        cheirality_weight=1.0,
    )

    assert int(result.visible_point_count) == 1
    assert float(result.mean_distance_px) == pytest.approx(1.0)
    assert float(result.invalid_depth_fraction) == pytest.approx(0.0)


def test_globally_zero_visibility_is_an_error() -> None:
    with pytest.raises(ValueError, match="GT-visible"):
        _consistency_loss(
            torch.zeros((1, 2, 2)),
            torch.zeros((1, 2, 2)),
            torch.ones((1, 2)),
            torch.zeros((1, 2), dtype=torch.bool),
        )


def test_negative_depth_is_finite_and_receives_cheirality() -> None:
    dense = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    pose = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    depth = torch.tensor([[-1.0]], requires_grad=True)
    result = _consistency_loss(
        dense,
        pose,
        depth,
        torch.tensor([[True]]),
        cheirality_weight=0.25,
    )

    expected_cheirality = torch.nn.functional.softplus(torch.tensor(2.2))
    torch.testing.assert_close(result.cheirality, expected_cheirality)
    assert float(result.invalid_depth_fraction) == pytest.approx(1.0)
    result.auxiliary.backward()
    for value in (dense, pose, depth):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


@pytest.mark.parametrize(
    ("gradient_flow", "dense_has_grad", "pose_has_grad", "depth_has_grad"),
    [
        ("both", True, True, True),
        ("stopgrad_pose", True, False, False),
        ("stopgrad_dense", False, True, True),
    ],
)
def test_gradient_flow_detaches_only_the_named_branch(
    gradient_flow: str,
    dense_has_grad: bool,
    pose_has_grad: bool,
    depth_has_grad: bool,
) -> None:
    dense = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    pose = torch.tensor([[[2.0, 1.0]]], requires_grad=True)
    depth = torch.tensor([[-1.0]], requires_grad=True)
    result = _consistency_loss(
        dense,
        pose,
        depth,
        torch.tensor([[True]]),
        cheirality_weight=0.25,
        gradient_flow=cast("CourtConsistencyGradientFlow", gradient_flow),
    )

    result.auxiliary.backward()

    assert (dense.grad is not None) is dense_has_grad
    assert (pose.grad is not None) is pose_has_grad
    assert (depth.grad is not None) is depth_has_grad


def test_soft_argmax_and_predicted_pose_both_receive_gradient() -> None:
    logits = torch.randn((1, 14, 4, 4), requires_grad=True)
    valid_mask = torch.ones_like(logits, dtype=torch.bool)
    dense_normalized = heatmaps_to_soft_argmax(
        logits,
        temperature=1.0,
        valid_mask=valid_mask,
    )
    dense_px = dense_normalized * logits.new_tensor([3.0, 3.0])
    raw_pose = torch.tensor(
        [
            [
                0.0,
                0.0,
                -10.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                math.log(20.0),
            ]
        ],
        requires_grad=True,
    )
    decoded = decode_pose10d_strict(raw_pose)
    canonical_points = canonical_semantic_court_points_batched(
        torch.arange(14, dtype=torch.long).unsqueeze(0)
    )
    projected = project_predicted_canonical_points(
        decoded,
        canonical_points,
        torch.tensor([[1.5, 1.5]]),
    )
    result = query_keypoint_pose_consistency_loss(
        dense_px,
        projected.points_xy,
        projected.depth_m,
        torch.tensor([[4, 4]], dtype=torch.long),
        torch.ones((1, 14), dtype=torch.bool),
        huber_delta=0.1,
        min_depth_m=0.1,
        depth_scale_m=1.0,
        cheirality_weight=0.1,
        gradient_flow="both",
    )

    result.auxiliary.backward()

    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert raw_pose.grad is not None and torch.isfinite(raw_pose.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
    assert torch.count_nonzero(raw_pose.grad[:, :3]) > 0
    assert torch.count_nonzero(raw_pose.grad[:, 3:9]) > 0
    assert torch.count_nonzero(raw_pose.grad[:, 9]) > 0


def test_bfloat16_autocast_full_pose_consistency_path_keeps_all_gradients() -> None:
    raw_pose = torch.tensor(
        [
            [
                1.3,
                -15.0,
                8.0,
                0.7,
                0.3,
                0.2,
                -0.1,
                0.8,
                0.4,
                math.log(20.0),
            ]
        ],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dense_points = torch.arange(28, dtype=torch.bfloat16).reshape(1, 14, 2)
    dense_points.requires_grad_()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        decoded = decode_pose10d_strict(raw_pose)
        canonical_points = canonical_semantic_court_points_batched(
            torch.arange(14, dtype=torch.long).unsqueeze(0),
            dtype=torch.bfloat16,
        )
        projected = project_predicted_canonical_points(
            decoded,
            canonical_points,
            torch.tensor([[8.0, 8.0]], dtype=torch.bfloat16),
        )
        result = query_keypoint_pose_consistency_loss(
            dense_points,
            projected.points_xy,
            projected.depth_m,
            torch.tensor([[16, 16]], dtype=torch.long),
            torch.ones((1, 14), dtype=torch.bool),
            huber_delta=0.1,
            min_depth_m=0.1,
            depth_scale_m=1.0,
            cheirality_weight=0.1,
            gradient_flow="both",
        )

    raw_gradient, dense_gradient = torch.autograd.grad(
        result.auxiliary,
        (raw_pose, dense_points),
    )
    assert decoded.rotation.dtype == torch.float32
    assert projected.points_xy.dtype == torch.float32
    assert result.auxiliary.dtype == torch.float32
    assert torch.isfinite(raw_gradient).all()
    assert torch.isfinite(dense_gradient).all()
    assert torch.all(raw_gradient != 0.0)
    assert torch.all(dense_gradient != 0.0)


@pytest.mark.parametrize(
    ("progress", "expected"),
    [(0.0, 0.0), (0.1, 0.0), (0.55, 1.0), (1.0, 2.0)],
)
def test_consistency_weight_zero_then_linearly_ramps(
    progress: float,
    expected: float,
) -> None:
    assert consistency_effective_weight(
        weight=2.0,
        warmup_fraction=0.1,
        progress=progress,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight": float("nan"), "warmup_fraction": 0.1, "progress": 0.5},
        {"weight": 1.0, "warmup_fraction": 1.0, "progress": 0.5},
        {"weight": 1.0, "warmup_fraction": 0.1, "progress": 1.1},
    ],
)
def test_consistency_weight_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        consistency_effective_weight(**kwargs)
