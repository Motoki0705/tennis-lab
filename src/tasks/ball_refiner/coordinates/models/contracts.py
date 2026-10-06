"""Coordinate input validation and flow-time features."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal(values: Tensor, width: int) -> Tensor:
    frequency = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=values.dtype) * (-math.log(10000) / width))
    phase = values[..., None] * frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def validate_input(coordinates: Tensor, missing: Tensor, dimensions: int) -> None:
    if coordinates.ndim != 3 or coordinates.shape[-1] != dimensions or coordinates.shape[1] < 1:
        raise ValueError(f"Coordinates must be B,T,{dimensions} with T>0")
    if missing.shape != coordinates.shape[:-1] or missing.dtype != torch.bool:
        raise ValueError("missing must be a B,T boolean tensor (true=missing)")
    if missing.device != coordinates.device or not coordinates.is_floating_point():
        raise ValueError("Coordinates and mask must share a device; coordinates must be floating point")
    if not torch.isfinite(coordinates[~missing]).all():
        raise ValueError("Observed coordinates must be finite")

