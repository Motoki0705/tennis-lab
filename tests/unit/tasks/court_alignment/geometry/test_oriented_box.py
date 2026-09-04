"""Tests for court OBB targets and axial geometry."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.court_alignment.geometry.court import (
    COURT_LENGTH_M,
    GroundCourtInstance,
    court_keypoints_for_instance,
)
from src.tasks.court_alignment.geometry.oriented_box import (
    COURT_SHORT_TO_LONG_RATIO,
    build_detr_court_targets,
    court_rotation_from_axial_vector,
    decode_raw_court_boxes,
    oriented_box_corners,
)


@pytest.mark.parametrize("rotation_deg", [0.0, 90.0, 179.0])
def test_kp14_target_recovers_axial_rotation_and_scale(rotation_deg: float) -> None:
    rotation = math.radians(rotation_deg)
    instance = GroundCourtInstance(0, (400.0, 400.0), rotation, 5.0)
    keypoints = court_keypoints_for_instance(instance)[None, None]
    visibility = torch.ones((1, 1, 14), dtype=torch.bool)

    target = build_detr_court_targets(keypoints, visibility, image_size=(800, 800))[0]

    assert target["labels"].tolist() == [0]
    assert target["boxes"].shape == (1, 4)
    torch.testing.assert_close(target["court_boxes"][0, :2], torch.tensor([0.5, 0.5]))
    assert float(target["court_boxes"][0, 2]) == pytest.approx(
        5.0 * COURT_LENGTH_M / 800.0,
        rel=1.0e-5,
    )
    recovered = court_rotation_from_axial_vector(target["court_boxes"][:, 3:])
    assert math.degrees(float(recovered[0])) == pytest.approx(rotation_deg, abs=1.0e-4)
    torch.testing.assert_close(
        torch.linalg.vector_norm(target["court_boxes"][:, 3:], dim=-1),
        torch.ones(1),
    )


def test_target_conversion_retains_multiple_instances_and_drops_invisible_padding() -> (
    None
):
    first = court_keypoints_for_instance(
        GroundCourtInstance(0, (100.0, 100.0), 0.2, 3.0)
    )
    second = court_keypoints_for_instance(
        GroundCourtInstance(1, (280.0, 260.0), 1.1, 3.5)
    )
    keypoints = torch.zeros((2, 3, 14, 2))
    visibility = torch.zeros((2, 3, 14), dtype=torch.bool)
    keypoints[0, :2] = torch.stack((first, second))
    visibility[0, :2] = True
    keypoints[1, 0] = first
    visibility[1, 0] = True
    visibility[1, 0, 3] = False

    targets = build_detr_court_targets(keypoints, visibility, image_size=(800, 800))

    assert len(targets) == 2
    assert targets[0]["labels"].shape == (2,)
    assert targets[0]["boxes"].shape == (2, 4)
    assert targets[0]["court_boxes"].shape == (2, 5)
    assert targets[1]["labels"].shape == (0,)
    assert targets[1]["boxes"].shape == (0, 4)
    assert targets[1]["court_boxes"].shape == (0, 5)


def test_rectangular_target_uses_per_axis_aabb_and_isotropic_long_side() -> None:
    instance = GroundCourtInstance(0, (200.0, 100.0), 0.0, 5.0)
    keypoints = court_keypoints_for_instance(instance)[None, None]
    visibility = torch.ones((1, 1, 14), dtype=torch.bool)

    target = build_detr_court_targets(
        keypoints,
        visibility,
        image_size=(400, 800),
    )[0]

    # x uses width=800, y uses height=400; physical lengths use max=800.
    torch.testing.assert_close(target["boxes"][0, :2], torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(
        target["boxes"][0, 2:],
        torch.tensor([5.0 * 10.97 / 800.0, 5.0 * 23.77 / 400.0]),
    )
    torch.testing.assert_close(target["court_boxes"][0, :2], torch.tensor([0.25, 0.25]))
    assert float(target["court_boxes"][0, 2]) == pytest.approx(
        5.0 * COURT_LENGTH_M / 800.0
    )


def test_raw_head_decode_is_finite_unit_axial_and_differentiable() -> None:
    raw = torch.tensor([[0.0, 3.0, 4.0], [1.0, 0.0, 0.0]], requires_grad=True)
    decoded = decode_raw_court_boxes(raw)

    torch.testing.assert_close(decoded[0], torch.tensor([0.5, 0.6, 0.8]))
    torch.testing.assert_close(decoded[1, 1:], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(
        torch.linalg.vector_norm(decoded[:, 1:], dim=-1), torch.ones(2)
    )
    decoded.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_oriented_corners_obey_fixed_regulation_ratio() -> None:
    center = torch.tensor([[100.0, 120.0]])
    long_side = torch.tensor([80.0])
    axial = torch.tensor([[-1.0, 0.0]])  # long edge parallel to image y

    corners = oriented_box_corners(center, long_side, axial)

    edge_lengths = torch.linalg.vector_norm(
        torch.roll(corners, shifts=-1, dims=1) - corners,
        dim=-1,
    )
    assert float(edge_lengths[0, 0]) == pytest.approx(80.0 * COURT_SHORT_TO_LONG_RATIO)
    assert float(edge_lengths[0, 1]) == pytest.approx(80.0)
