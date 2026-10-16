"""Prepare augmented, triangulated inputs and full-rally event targets."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.data.augmentation.pipeline import augment_observations
from src.tasks.ball_refiner_3d.data.schema import (
    BoolArray,
    CorruptedTrajectory,
    FloatArray,
    PreparedRally,
    Rally,
)
from src.tasks.ball_refiner_3d.data.targets.events import gaussian_event_target
from src.tasks.ball_refiner_3d.model_io.adapters import normalization
from src.tasks.ball_refiner_3d.physics.targets import physics_targets
from src.utils.geometry.triangulation import triangulate_multiview


def prepare(
    rallies: list[Rally],
    dimensions: int,
    config: CorruptionConfig,
    seed: int,
    *,
    event_sigma_frames: float,
) -> list[PreparedRally]:
    scale, offset = normalization(dimensions)
    output = []
    for rally in rallies:
        corruption_seed = int(
            np.random.SeedSequence([seed, rally.index]).generate_state(1)[0]
        )
        corrupted = corrupt_trajectory(
            rally.uv,
            rally.visible,
            rally.projection,
            rally.events,
            config=config,
            seed=corruption_seed,
        )
        inputs = corrupted.xyz_m[None]
        missing = corrupted.missing_3d[None]
        truth = rally.xyz[None]
        normalized = np.where(missing[..., None], 0, inputs / scale - offset).astype(
            np.float32
        )
        output.append(
            PreparedRally(
                rally,
                corrupted,
                normalized,
                missing,
                (truth / scale - offset).astype(np.float32),
                gaussian_event_target(rally.events, event_sigma_frames)[None],
                physics_targets(rally.physics, rally.xyz),
            )
        )
    return output


def corrupt_trajectory(
    uv_px: FloatArray,
    visible: BoolArray,
    projections: NDArray[np.floating],
    events: NDArray[np.integer],
    *,
    config: CorruptionConfig,
    seed: int,
    noise_enabled: bool = True,
) -> CorruptedTrajectory:
    observations = augment_observations(
        uv_px, visible, events, config=config, seed=seed, noise_enabled=noise_enabled
    )
    result = triangulate_multiview(
        observations.uv_px.transpose(1, 0, 2),
        (~observations.missing).T.astype(np.float64),
        projections,
        min_score=0.5,
        refinement_steps=config.triangulation_steps,
    )
    return CorruptedTrajectory(
        observations.uv_px,
        observations.missing,
        np.where(result.valid[:, None], result.points, 0).astype(np.float32),
        ~result.valid,
        observations.intervals,
        observations.isolated,
        observations.noise_px,
    )
