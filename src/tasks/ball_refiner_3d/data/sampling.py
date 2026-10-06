"""Uniform rally and temporal window sampling."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner_3d.data.schema import PreparedRally


def sample_batch(
    data: list[PreparedRally],
    batch_size: int,
    length: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    coords, masks, targets, event_targets = [], [], [], []
    # Equal rally sampling. Events supervise a separate output, never input features.
    for index in rng.integers(len(data), size=batch_size):
        rally = data[index]
        if rally.coordinates.shape[1] < length:
            raise ValueError("Rally shorter than the configured training window")
        view = int(rng.integers(len(rally.coordinates)))
        start = int(rng.integers(rally.coordinates.shape[1] - length + 1))
        coords.append(rally.coordinates[view, start : start + length])
        masks.append(rally.missing[view, start : start + length])
        targets.append(rally.target[view, start : start + length])
        event_targets.append(rally.event_target[view, start : start + length])
    return tuple(
        torch.from_numpy(np.stack(values)).to(device)
        for values in (coords, masks, targets, event_targets)
    )
