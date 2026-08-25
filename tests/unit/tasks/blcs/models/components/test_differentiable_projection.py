"""Fixed normalization coverage for differentiable BLCS projection."""

from __future__ import annotations

import torch

from src.tasks.blcs.models.components.differentiable_projection import (
    DifferentiableProjection,
)


def test_projection_denormalizes_with_fixed_isotropic_scale_and_backpropagates() -> (
    None
):
    projector = DifferentiableProjection()
    position_norm = torch.tensor([[[1.0, 0.0, 0.0]]], requires_grad=True)
    rotation = torch.eye(3).view(1, 1, 3, 3)
    camera_center = torch.tensor([[[0.0, 0.0, -11.885]]])
    scalar = torch.ones(1, 1)

    uv, in_front = projector(
        position_norm,
        rotation,
        camera_center,
        scalar,
        torch.zeros_like(scalar),
        torch.zeros_like(scalar),
        scalar,
        scalar,
    )

    torch.testing.assert_close(uv, torch.tensor([[[[1.0, 0.0]]]]))
    assert in_front.all()
    uv.sum().backward()
    assert position_norm.grad is not None
    assert torch.isfinite(position_norm.grad).all()
