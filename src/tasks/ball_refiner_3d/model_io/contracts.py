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


class PhysicsOutput(NamedTuple):
    """Network-unit field ``(B,4)``, surface logits ``(B,3)`` and, when a
    segmentation was given, segment initial states ``(B,S,9)``."""

    field: Tensor
    surface_logits: Tensor
    segment_states: Tensor | None


class RefinerOutput(NamedTuple):
    coordinates: Tensor
    event_logits: Tensor
    physics: PhysicsOutput | None = None

    @property
    def event_probability(self) -> Tensor:
        return self.event_logits.softmax(dim=-1)[..., 1]


class PhysicsPrediction(NamedTuple):
    """Predicted physics of whole sequences ``V``.

    ``field (V,4)`` and ``segment_states (V,S,9)`` are in network units (rows
    beyond a sequence's segments are zero), ``segment (V,T)`` holds the flight
    labels the states belong to, and ``integrated (V,T,3)`` is their integrated
    trajectory in the same units as the predicted coordinates.
    """

    field: Tensor
    surface_probability: Tensor
    segment: Tensor
    segment_states: Tensor
    integrated: Tensor


class RefinerPrediction(NamedTuple):
    coordinates: Tensor
    event_probability: Tensor
    physics: PhysicsPrediction | None = None
