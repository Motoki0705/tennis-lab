"""Differentiable batched pinhole projection for world-space points."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentiablePinholeProjection(nn.Module):
    """Project arbitrary world-point layouts into normalized camera UV.

    World points use shape ``(B, ..., 3)`` and camera parameters use a separate
    view axis ``(B, V, ...)``. The returned tensors therefore have shapes
    ``(B, V, ..., 2)`` and ``(B, V, ...)``. Camera parameters are fixed inputs;
    gradients propagate only through ``world_points``.
    """

    def __init__(self, *, depth_eps: float = 1e-6) -> None:
        super().__init__()
        if depth_eps <= 0.0:
            raise ValueError("depth_eps must be positive.")
        self.depth_eps = float(depth_eps)

    @staticmethod
    def validate_inputs(
        world_points: Tensor,
        camera_R: Tensor,
        camera_C: Tensor,
        camera_f: Tensor,
        camera_cx: Tensor,
        camera_cy: Tensor,
        camera_w: Tensor,
        camera_h: Tensor,
    ) -> tuple[int, int, int]:
        """Validate tensor shapes and return projection layout dimensions."""
        if world_points.ndim < 2 or world_points.shape[-1] != 3:
            raise ValueError(
                "world_points must have shape (B, ..., 3), got "
                f"{tuple(world_points.shape)}."
            )
        if not world_points.is_floating_point():
            raise TypeError("world_points must use a floating-point dtype.")
        if camera_R.ndim != 4 or camera_R.shape[-2:] != (3, 3):
            raise ValueError(
                "camera_R must have shape (B, V, 3, 3), got "
                f"{tuple(camera_R.shape)}."
            )

        batch_size, num_views = camera_R.shape[:2]
        expected_shapes = {
            "camera_C": (batch_size, num_views, 3),
            "camera_f": (batch_size, num_views),
            "camera_cx": (batch_size, num_views),
            "camera_cy": (batch_size, num_views),
            "camera_w": (batch_size, num_views),
            "camera_h": (batch_size, num_views),
        }
        camera_tensors = {
            "camera_C": camera_C,
            "camera_f": camera_f,
            "camera_cx": camera_cx,
            "camera_cy": camera_cy,
            "camera_w": camera_w,
            "camera_h": camera_h,
        }
        for name, expected in expected_shapes.items():
            actual = camera_tensors[name]
            if tuple(actual.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(actual.shape)}."
                )
        if world_points.shape[0] != batch_size:
            raise ValueError(
                "world_points and cameras must share the batch axis, got "
                f"{world_points.shape[0]} and {batch_size}."
            )
        point_rank = world_points.ndim - 2
        return batch_size, num_views, point_rank

    def forward(
        self,
        world_points: Tensor,
        camera_R: Tensor,
        camera_C: Tensor,
        camera_f: Tensor,
        camera_cx: Tensor,
        camera_cy: Tensor,
        camera_w: Tensor,
        camera_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return normalized UV and a positive-depth mask for each camera."""
        batch_size = world_points.shape[0]
        num_views = camera_R.shape[1]
        point_rank = world_points.ndim - 2
        device = world_points.device
        dtype = world_points.dtype

        def fixed(value: Tensor) -> Tensor:
            return value.detach().to(device=device, dtype=dtype)

        view_shape = (batch_size, num_views, *([1] * point_rank))
        center = fixed(camera_C).view(*view_shape, 3)
        rotation = fixed(camera_R)
        relative = world_points.unsqueeze(1) - center
        camera_points = torch.einsum(
            "bv...j,bvkj->bv...k",
            relative,
            rotation,
        )

        depth = camera_points[..., 2]
        in_front = depth > self.depth_eps
        safe_depth = torch.where(in_front, depth, torch.ones_like(depth))

        focal = fixed(camera_f).view(view_shape)
        center_x = fixed(camera_cx).view(view_shape)
        center_y = fixed(camera_cy).view(view_shape)
        width = fixed(camera_w).view(view_shape).clamp_min(1.0)
        height = fixed(camera_h).view(view_shape).clamp_min(1.0)

        u = (focal * camera_points[..., 0] / safe_depth + center_x) / width
        v = (focal * camera_points[..., 1] / safe_depth + center_y) / height
        return torch.stack((u, v), dim=-1), in_front


__all__ = ["DifferentiablePinholeProjection"]
