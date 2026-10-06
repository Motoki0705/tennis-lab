"""Coordinate inputs and typed model outputs."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


def validate_input(coordinates: Tensor, missing: Tensor, dimensions: int) -> None:
    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] != dimensions
        or coordinates.shape[1] < 1
    ):
        raise ValueError(f"Coordinates must be B,T,{dimensions} with T>0")
    if missing.shape != coordinates.shape[:-1] or missing.dtype != torch.bool:
        raise ValueError("missing must be a B,T boolean tensor (true=missing)")
    if missing.device != coordinates.device or not coordinates.is_floating_point():
        raise ValueError(
            "Coordinates and mask must share a device; coordinates must be floating point"
        )
    if not torch.isfinite(coordinates[~missing]).all():
        raise ValueError("Observed coordinates must be finite")


class RefinerOutput(NamedTuple):
    coordinates: Tensor
    event_logits: Tensor

    @property
    def event_probability(self) -> Tensor:
        return self.event_logits.softmax(dim=-1)[..., 1]


class RefinerPrediction(NamedTuple):
    coordinates: Tensor
    event_probability: Tensor
