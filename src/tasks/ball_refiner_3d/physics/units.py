"""Network units of the physical parameters the refiner predicts.

Each quantity is mapped to values of order one over the regenerated dataset:
wind is sampled within +-3 m/s, k_drag in [0.005, 0.02] and k_magnus in
[0.0005, 0.002] (log units around the range centre), velocity p99 is about
(11, 23, 11) m/s and spin p99 about (350, 150, 200) rad/s.  Positions use the
coordinate normalization of the model input.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.utils.physics.ball.surfaces import SURFACE_NAMES

FIELD_CHANNELS = 4  # wind_x, wind_y, log k_drag, log k_magnus
STATE_CHANNELS = 9  # position, velocity, spin
SURFACES = SURFACE_NAMES

WIND_SCALE_MPS = 3.0
K_DRAG_REFERENCE = 0.01
K_MAGNUS_REFERENCE = 0.001
LOG_COEFFICIENT_SCALE = math.log(2.0)
POSITION_SCALE_M = tuple(float(v) for v in normalization(3)[0])
if normalization(3)[1].any():
    raise RuntimeError("State positions assume zero-offset coordinate normalization")
VELOCITY_SCALE_MPS = (10.0, 25.0, 10.0)
SPIN_SCALE_RADPS = (300.0, 150.0, 200.0)


class Field(NamedTuple):
    """Physical field ``wind (...,3)`` (no vertical wind), ``k_drag``, ``k_magnus``."""

    wind: Tensor
    k_drag: Tensor
    k_magnus: Tensor


class State(NamedTuple):
    """Physical flight state, each ``(...,3)``."""

    position: Tensor
    velocity: Tensor
    spin: Tensor


def _scale(values: tuple[float, ...], like: Tensor) -> Tensor:
    return like.new_tensor(values)


def encode_field(field: Field) -> Tensor:
    if not torch.all(field.wind[..., 2] == 0):
        raise ValueError("The refiner field has horizontal wind only")
    if not (torch.all(field.k_drag > 0) and torch.all(field.k_magnus > 0)):
        raise ValueError("Coefficients must be positive")
    return torch.cat(
        (
            field.wind[..., :2] / WIND_SCALE_MPS,
            ((field.k_drag.log() - math.log(K_DRAG_REFERENCE)) / LOG_COEFFICIENT_SCALE)[
                ..., None
            ],
            (
                (field.k_magnus.log() - math.log(K_MAGNUS_REFERENCE))
                / LOG_COEFFICIENT_SCALE
            )[..., None],
        ),
        dim=-1,
    )


def decode_field(values: Tensor) -> Field:
    if values.shape[-1] != FIELD_CHANNELS:
        raise ValueError(f"Field values must have {FIELD_CHANNELS} channels")
    wind_xy = values[..., :2] * WIND_SCALE_MPS
    return Field(
        torch.cat((wind_xy, torch.zeros_like(wind_xy[..., :1])), dim=-1),
        torch.exp(values[..., 2] * LOG_COEFFICIENT_SCALE) * K_DRAG_REFERENCE,
        torch.exp(values[..., 3] * LOG_COEFFICIENT_SCALE) * K_MAGNUS_REFERENCE,
    )


PHYSICAL_FIELD_COLUMNS = ("wind_x_mps", "wind_y_mps", "k_drag", "k_magnus")


def physical_field(field: Field) -> Tensor:
    """``(...,4)`` reported field in :data:`PHYSICAL_FIELD_COLUMNS` order."""
    return torch.cat(
        (field.wind[..., :2], field.k_drag[..., None], field.k_magnus[..., None]),
        dim=-1,
    )


def encode_state(state: State) -> Tensor:
    like = state.position
    return torch.cat(
        (
            state.position / _scale(POSITION_SCALE_M, like),
            state.velocity / _scale(VELOCITY_SCALE_MPS, like),
            state.spin / _scale(SPIN_SCALE_RADPS, like),
        ),
        dim=-1,
    )


def decode_state(values: Tensor) -> State:
    if values.shape[-1] != STATE_CHANNELS:
        raise ValueError(f"State values must have {STATE_CHANNELS} channels")
    return State(
        values[..., 0:3] * _scale(POSITION_SCALE_M, values),
        values[..., 3:6] * _scale(VELOCITY_SCALE_MPS, values),
        values[..., 6:9] * _scale(SPIN_SCALE_RADPS, values),
    )
