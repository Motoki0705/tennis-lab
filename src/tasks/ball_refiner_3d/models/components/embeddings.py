"""Input and flow-time embeddings; no sampling or loss calculation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def coordinate_features(coordinates: Tensor, missing: Tensor) -> Tensor:
    clean = torch.where(missing[..., None], 0, coordinates)
    return torch.cat((clean, missing[..., None].to(coordinates.dtype)), dim=-1)


def sinusoidal(values: Tensor, width: int) -> Tensor:
    frequency = torch.exp(
        torch.arange(0, width, 2, device=values.device, dtype=values.dtype)
        * (-math.log(10000) / width)
    )
    phase = values[..., None] * frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def time_embedding(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
