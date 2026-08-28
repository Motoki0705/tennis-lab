"""Tests for the shared differentiable pinhole projection."""

from __future__ import annotations

import pytest
import torch

from src.utils.projection.camera_projector import Camera, project_points
from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)


def _camera_batch() -> tuple[torch.Tensor, ...]:
    rotation = torch.eye(3).view(1, 1, 3, 3)
    center = torch.tensor([[[0.0, 0.0, -10.0]]])
    focal = torch.tensor([[800.0]])
    center_x = torch.tensor([[640.0]])
    center_y = torch.tensor([[360.0]])
    width = torch.tensor([[1280.0]])
    height = torch.tensor([[720.0]])
    return rotation, center, focal, center_x, center_y, width, height


def test_projection_matches_camera_projector_numerically() -> None:
    points = torch.tensor([[[1.0, 2.0, 0.5], [-2.0, 1.0, 1.5]]])
    rotation, center, focal, center_x, center_y, width, height = _camera_batch()
    uv, in_front = DifferentiablePinholeProjection()(
        points,
        rotation,
        center,
        focal,
        center_x,
        center_y,
        width,
        height,
    )

    camera = Camera(
        C=center[0, 0],
        R=rotation[0, 0],
        f=float(focal[0, 0]),
        cx=float(center_x[0, 0]),
        cy=float(center_y[0, 0]),
        w=int(width[0, 0]),
        h=int(height[0, 0]),
    )
    expected_uv, expected_front = project_points(camera, points[0])
    expected_uv = expected_uv / torch.tensor([camera.w, camera.h])

    torch.testing.assert_close(uv[0, 0], expected_uv)
    torch.testing.assert_close(in_front[0, 0], expected_front)


def test_projection_preserves_arbitrary_point_axes_and_backpropagates() -> None:
    points = torch.linspace(-0.5, 0.5, 2 * 3 * 3).reshape(1, 2, 3, 3)
    points.requires_grad_()
    rotation, center, focal, center_x, center_y, width, height = _camera_batch()
    rotation.requires_grad_()

    uv, in_front = DifferentiablePinholeProjection()(
        points,
        rotation,
        center,
        focal,
        center_x,
        center_y,
        width,
        height,
    )

    assert uv.shape == (1, 1, 2, 3, 2)
    assert in_front.shape == (1, 1, 2, 3)
    assert in_front.all()
    uv.sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert rotation.grad is None


def test_projection_rejects_mismatched_camera_shapes() -> None:
    points = torch.zeros(2, 3, 3)
    rotation, center, focal, center_x, center_y, width, height = _camera_batch()
    projector = DifferentiablePinholeProjection()

    with pytest.raises(ValueError, match="share the batch axis"):
        projector.validate_inputs(
            points,
            rotation,
            center,
            focal,
            center_x,
            center_y,
            width,
            height,
        )
