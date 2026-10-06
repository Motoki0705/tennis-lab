"""Shared rally storage and prepared training sample contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.utils.physics.ball.record import BallPhysicsRecord

FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]
SCHEMA = "ball_refiner.single_object.v2"


@dataclass(frozen=True)
class CorruptedTrajectory:
    uv_px: FloatArray  # V,T,2; zero where missing
    missing_2d: BoolArray  # V,T; true means missing
    xyz_m: FloatArray  # T,3; zero where missing
    missing_3d: BoolArray
    intervals: NDArray[np.int64]  # selected event, inclusive start, exclusive end
    isolated: BoolArray  # V,T; excludes event gaps and out-of-frame
    noise_px: FloatArray  # before masking, retained for audits only


@dataclass(frozen=True)
class Rally:
    index: int
    name: str
    split: str
    xyz: NDArray[np.float32]
    uv: NDArray[np.float32]
    visible: NDArray[np.bool_]
    projection: NDArray[np.float64]
    events: NDArray[np.uint8]  # EVENT_BITS per frame, derived from ``physics``
    time: NDArray[np.float64]
    physics: BallPhysicsRecord


@dataclass(frozen=True)
class PreparedRally:
    source: Rally
    corrupted: CorruptedTrajectory
    coordinates: NDArray[np.float32]  # V,T,D (3D uses V=1)
    missing: NDArray[np.bool_]
    target: NDArray[np.float32]
    event_target: NDArray[np.float32]


@dataclass(frozen=True)
class AugmentedObservations:
    uv_px: FloatArray
    missing: BoolArray
    intervals: NDArray[np.int64]
    isolated: BoolArray
    noise_px: FloatArray
