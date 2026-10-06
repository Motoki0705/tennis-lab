"""Generate degraded observations and triangulate the 3D input."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.data.augmentation.noise import coordinate_noise
from src.tasks.ball_refiner_3d.data.augmentation.occlusion import sample_occlusion
from src.tasks.ball_refiner_3d.data.schema import (
    AugmentedObservations,
    BoolArray,
    FloatArray,
)


def augment_observations(
    uv_px: FloatArray,
    visible: BoolArray,
    events: NDArray[np.integer],
    *,
    config: CorruptionConfig,
    seed: int,
    noise_enabled: bool = True,
) -> AugmentedObservations:
    """Synchronize event gaps across views; sample isolated drops independently.

    Random streams are separate, so rate ablations have nested event selection
    and identical noise/widths/isolated proposals. Overlapping gaps are unioned.
    Event metadata is never returned as a model feature.
    """
    if uv_px.ndim != 3 or uv_px.shape[-1] != 2 or visible.shape != uv_px.shape[:-1]:
        raise ValueError("Require V,T,2 coordinates and V,T visibility")
    frames = uv_px.shape[1]
    if events.shape != (frames,) or not np.isfinite(uv_px).all():
        raise ValueError("Require finite coordinates and one event bitmask per frame")
    event_rng, noise_rng, isolated_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    missing, isolated, intervals = sample_occlusion(
        visible, events, config, event_rng, isolated_rng
    )
    noise = (
        coordinate_noise(visible.shape, config, noise_rng)
        if noise_enabled
        else np.zeros_like(uv_px)
    )
    observation = np.where(missing[..., None], 0, uv_px + noise).astype(np.float32)
    return AugmentedObservations(observation, missing, intervals, isolated, noise)
