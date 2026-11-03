"""Tests for the shared differentiable pinhole projection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.utils.projection.differentiable_projection import (
    DifferentiablePinholeProjection,
)


def _projection_inputs() -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Two batches, three distinct cameras each, and two unequal point axes."""
    points = torch.tensor(
        [
            [
                [[1.2, -2.3, 0.5], [-0.7, 1.4, 2.6], [3.1, 0.6, -1.3]],
                [[-2.4, 0.1, 1.7], [0.8, -1.9, 3.2], [2.3, 2.8, -0.4]],
            ],
            [
                [[-1.6, 2.1, 1.3], [2.7, -0.4, -0.8], [0.3, 1.8, 2.9]],
                [[3.4, -2.6, 0.2], [-0.9, -1.2, 3.7], [1.1, 0.7, -1.5]],
            ],
        ],
        dtype=torch.float64,
    )
    angles = np.array(
        [
            [[0.2, -0.3, 0.1], [-0.4, 0.1, -0.2], [0.3, 0.2, -0.1]],
            [[-0.1, 0.3, 0.2], [0.4, -0.2, 0.3], [-0.3, -0.1, -0.4]],
        ]
    )
    rotations = np.empty((2, 3, 3, 3))
    for batch, view in np.ndindex(2, 3):
        x, y, z = angles[batch, view]
        rx = np.array(
            [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
        )
        ry = np.array(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        rz = np.array(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
        )
        rotations[batch, view] = rz @ ry @ rx
    cameras = (
        rotations,
        [
            [[1.3, -2.1, -8.0], [-2.8, 0.6, -11.0], [0.4, 3.7, -7.5]],
            [[-1.9, 1.1, -12.0], [2.6, -3.4, -9.0], [3.2, 0.8, -10.5]],
        ],
        [[730, 1130, 570], [920, 610, 1400]],
        [[311, 720, 277], [615, 477, 850]],
        [[201, 460, 319], [333, 417, 500]],
        [[960, 1600, 800], [1280, 1024, 1920]],
        [[540, 900, 600], [720, 768, 1080]],
    )
    return points, tuple(torch.tensor(value, dtype=torch.float64) for value in cameras)


def _numpy_projection(
    points: NDArray[np.float64], cameras: tuple[torch.Tensor, ...]
) -> NDArray[np.float64]:
    """Scalar camera/point reference, independent of torch's batched projection."""
    rotation, center, focal, cx, cy, width, height = (
        value.detach().numpy() for value in cameras
    )
    uv = np.empty((2, 3, 2, 3, 2))
    for batch, view, row, col in np.ndindex(2, 3, 2, 3):
        q = rotation[batch, view] @ (points[batch, row, col] - center[batch, view])
        assert q[2] > 1e-6  # This reference is only used for front-facing points.
        intrinsic = np.array(
            [
                [focal[batch, view], 0, cx[batch, view]],
                [0, focal[batch, view], cy[batch, view]],
                [0, 0, 1],
            ]
        )
        pixel = intrinsic @ q
        uv[batch, view, row, col] = (
            pixel[:2] / pixel[2] / [width[batch, view], height[batch, view]]
        )
    return uv


def test_projection_matches_independent_multibatch_multiview_values() -> None:
    points, cameras = _projection_inputs()
    uv, in_front = DifferentiablePinholeProjection()(points, *cameras)

    torch.testing.assert_close(
        uv,
        torch.from_numpy(_numpy_projection(points.numpy(), cameras)),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(in_front, torch.ones(2, 3, 2, 3, dtype=torch.bool))


def test_projection_gradient_matches_independent_float64_finite_differences() -> None:
    points, cameras = _projection_inputs()
    points.requires_grad_()
    for camera in cameras:
        camera.requires_grad_()
    uv, _ = DifferentiablePinholeProjection()(points, *cameras)
    weights = np.linspace(-0.7, 1.3, uv.numel()).reshape(tuple(uv.shape))
    (uv * torch.from_numpy(weights)).sum().backward()

    step = 1e-5
    expected_gradient = np.empty(tuple(points.shape))
    for index in np.ndindex(*points.shape):
        plus = points.detach().numpy().copy()
        minus = plus.copy()
        plus[index] += step
        minus[index] -= step
        difference = _numpy_projection(plus, cameras) - _numpy_projection(
            minus, cameras
        )
        expected_gradient[index] = (difference * weights).sum() / (2 * step)
    assert points.grad is not None
    torch.testing.assert_close(
        points.grad, torch.from_numpy(expected_gradient), rtol=1e-6, atol=1e-9
    )
    assert all(camera.grad is None for camera in cameras)


def test_depth_mask_excludes_eps_and_back_points_from_downstream_gradient() -> None:
    eps = 1e-6
    delta = eps * 1e-3
    depths = [-1.0, 0.0, eps - delta, eps, eps + delta]
    points = torch.tensor(
        [[[eps / 2, -eps / 4, depth] for depth in depths]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cameras = (
        torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3),
        torch.zeros(1, 1, 3, dtype=torch.float64),
        *(
            torch.tensor([[value]], dtype=torch.float64)
            for value in (800, 640, 360, 1280, 720)
        ),
    )
    uv, in_front = DifferentiablePinholeProjection(depth_eps=eps)(points, *cameras)
    torch.testing.assert_close(
        in_front, torch.tensor([[[False, False, False, False, True]]])
    )
    # Invalid UV is unspecified; downstream selects only positive-depth entries.
    uv[in_front].sum().backward()
    assert points.grad is not None
    torch.testing.assert_close(
        points.grad[0, :4], torch.zeros(4, 3, dtype=torch.float64)
    )
    assert points.grad[0, 4].norm() > 0


def test_projection_rejects_mismatched_camera_shapes() -> None:
    points = torch.zeros(1, 3, 3)
    _, cameras = _projection_inputs()
    projector = DifferentiablePinholeProjection()

    with pytest.raises(ValueError, match="share the batch axis"):
        projector.validate_inputs(
            points,
            *cameras,
        )
