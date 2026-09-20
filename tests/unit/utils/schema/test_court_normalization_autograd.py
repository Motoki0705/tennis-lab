"""Coordinate helpers retain differentiable Tensor outputs and floating dtype."""

import torch

from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
    normalize_court_position,
    normalize_court_velocity,
)


def test_coordinate_conversions_preserve_autograd_on_noncontiguous_tensor() -> None:
    for normalize, denormalize in (
        (normalize_court_position, denormalize_court_position),
        (normalize_court_velocity, denormalize_court_velocity),
    ):
        source = torch.arange(18, dtype=torch.float64).reshape(3, 6).requires_grad_()
        coordinates = source.T
        assert not coordinates.is_contiguous()
        normalized = normalize(coordinates)
        assert normalized.dtype == source.dtype
        torch.testing.assert_close(normalized, coordinates / source.new_tensor(COURT_COORD_SCALE_XYZ))
        restored = denormalize(normalized)
        torch.testing.assert_close(restored, coordinates)
        restored.sum().backward()
        torch.testing.assert_close(source.grad, torch.ones_like(source))
