"""Long-sequence inference with nearest-center ownership."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.ball_refiner_3d.inference.flow_sampler import predict_window
from src.tasks.ball_refiner_3d.model_io.contracts import (
    RefinerPrediction,
    validate_input,
)
from src.tasks.ball_refiner_3d.model_io.factory import RefinerModel


def window_starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
    """Backfill the tail with real frames; a short clip is an explicit error."""
    if length < 1 or not 1 <= stride <= length or frames < length:
        raise ValueError("Windows require frames >= length >= stride >= 1; no padding")
    starts = list(range(0, frames - length + 1, stride))
    if starts[-1] != frames - length:
        starts.append(frames - length)
    return tuple(starts)


def window_owners(frames: int, starts: Sequence[int], length: int) -> NDArray[np.int64]:
    """One prediction per frame: nearest centre, ties resolved by earlier start."""
    owners: NDArray[np.int64] = np.full(frames, -1, dtype=np.int64)
    distances = np.full(frames, np.inf)
    if not starts or list(starts) != sorted(set(starts)):
        raise ValueError("Window starts must be nonempty, sorted and unique")
    for index, start in enumerate(starts):
        if start < 0 or start + length > frames:
            raise ValueError("Window exceeds the source timeline")
        rows = np.arange(start, start + length)
        distance = np.abs(rows - (start + (length - 1) / 2))
        take = distance < distances[rows]
        owners[rows[take]] = index
        distances[rows[take]] = distance[take]
    if (owners < 0).any():
        raise ValueError("Window policy leaves source frames uncovered")
    return owners


@torch.no_grad()  # type: ignore[untyped-decorator]
def predict_normalized(
    model: RefinerModel,
    coordinates: Tensor,
    missing: Tensor,
    *,
    batch_size: int,
    seed: int,
) -> RefinerPrediction:
    """V,T,D -> V,T,D. Real windows, nearest-center ownership, no mode averaging."""
    validate_input(coordinates, missing, model.config.dimensions)
    if batch_size < 1 or seed < 0:
        raise ValueError("Require positive batch size and nonnegative seed")
    frames = coordinates.shape[1]
    length = min(frames, model.config.window_length)
    starts = window_starts(frames, length, max(1, length // 2))
    owners = window_owners(frames, starts, length)
    output = torch.empty_like(coordinates)
    events = torch.empty_like(coordinates[..., 0])
    generator = torch.Generator(device=coordinates.device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for view in range(len(coordinates)):
            for first in range(0, len(starts), batch_size):
                selected = starts[first : first + batch_size]
                inputs = torch.stack(
                    [coordinates[view, start : start + length] for start in selected]
                )
                mask = torch.stack(
                    [missing[view, start : start + length] for start in selected]
                )
                prediction = predict_window(model, inputs, mask, generator=generator)
                for local, start in enumerate(selected):
                    take = np.flatnonzero(
                        owners[start : start + length] == first + local
                    )
                    indices = torch.from_numpy(take).to(coordinates.device)
                    output[view, indices + start] = prediction.coordinates[
                        local, indices
                    ]
                    events[view, indices + start] = prediction.event_probability[
                        local, indices
                    ]
    finally:
        model.train(was_training)
    if not torch.isfinite(output).all() or not torch.isfinite(events).all():
        raise ValueError("Refiner produced nonfinite coordinates")
    return RefinerPrediction(output, events)
