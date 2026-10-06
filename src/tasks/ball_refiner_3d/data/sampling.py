"""Uniform rally sampling and variable-length, padded training windows."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.ball_refiner_3d.configuration.data import WindowConfig
from src.tasks.ball_refiner_3d.data.schema import PreparedRally
from src.tasks.ball_refiner_3d.physics.targets import local_segments
from src.tasks.ball_refiner_3d.physics.units import FIELD_CHANNELS, STATE_CHANNELS


def window_length(frames: int, config: WindowConfig, rng: np.random.Generator) -> int:
    """``config.length``, or with ``long_probability`` a uniform longer length."""
    if frames < config.length:
        raise ValueError("Rally shorter than the configured training window")
    longest = min(frames, config.max_length)
    if rng.random() < config.long_probability:
        return int(rng.integers(config.length, longest + 1))
    return config.length


def sample_batch(
    data: list[PreparedRally],
    batch_size: int,
    config: WindowConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """Windows padded to the longest one; ``padding`` frames are also missing.

    Segment labels restart at 0 in each window; a segment cut by the window
    start is supervised by its state at the window's first frame.
    """
    windows = []
    # Equal rally sampling. Events supervise a separate output, never input features.
    for index in rng.integers(len(data), size=batch_size):
        rally = data[index]
        frames = rally.coordinates.shape[1]
        view = int(rng.integers(len(rally.coordinates)))
        length = window_length(frames, config, rng)
        start = int(rng.integers(frames - length + 1))
        windows.append((rally, view, start, length))
    longest = max(length for *_, length in windows)
    segments = [
        local_segments(rally.physics.segment[start : start + length])
        for rally, _, start, length in windows
    ]
    count = max(int(labels.max()) + 1 for labels in segments)
    batch: dict[str, NDArray[Any]] = {
        "coordinates": np.zeros((batch_size, longest, 3), np.float32),
        "missing": np.ones((batch_size, longest), bool),
        "padding": np.ones((batch_size, longest), bool),
        "target": np.zeros((batch_size, longest, 3), np.float32),
        "event_target": np.zeros((batch_size, longest), np.float32),
        "segment": np.full((batch_size, longest), -1, np.int64),
        "segment_target": np.zeros((batch_size, count, STATE_CHANNELS), np.float32),
        "field_target": np.zeros((batch_size, FIELD_CHANNELS), np.float32),
        "surface_target": np.zeros(batch_size, np.int64),
    }
    for row, ((rally, view, start, length), labels) in enumerate(
        zip(windows, segments, strict=True)
    ):
        frames = slice(start, start + length)
        batch["coordinates"][row, :length] = rally.coordinates[view, frames]
        batch["missing"][row, :length] = rally.missing[view, frames]
        batch["padding"][row, :length] = False
        batch["target"][row, :length] = rally.target[view, frames]
        batch["event_target"][row, :length] = rally.event_target[view, frames]
        batch["segment"][row, :length] = labels
        firsts = np.flatnonzero(np.diff(labels, prepend=-1))
        batch["segment_target"][row, : len(firsts)] = rally.physics.state[
            start + firsts
        ]
        batch["field_target"][row] = rally.physics.field
        batch["surface_target"][row] = rally.physics.surface
    return {key: torch.from_numpy(value).to(device) for key, value in batch.items()}
