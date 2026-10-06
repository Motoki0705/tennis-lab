"""Translate BLCS physics and event ownership into refiner rallies."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.configuration.generation import (
    CameraSampling,
    GenerationConfig,
)
from src.tasks.ball_refiner_3d.generate_dataset.cameras import sample_visible_cameras
from src.tasks.blcs.generate_dataset.physics_retry import (
    generate_with_bounded_physics_resampling,
)
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyResult,
    RallySimulator,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import (
    BASELINE_CLEAR,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
)


def event_frames(
    result: RallyResult, stride: int = 1, frames: int | None = None
) -> NDArray[np.uint8]:
    """Discard hypothetical later bounces after a shot was actually returned."""
    events: NDArray[np.uint8] = np.zeros(
        frames if frames is not None else len(result.trajectory[::stride]),
        dtype=np.uint8,
    )
    for index, shot in enumerate(result.shot_events):
        end = (
            result.shot_events[index + 1].t_start
            if index + 1 < len(result.shot_events)
            else len(result.trajectory)
        )
        if shot.t_return >= 0:
            end = min(end, shot.t_return + 1)
        start_frame = int(np.floor(shot.t_start / stride + 0.5))
        if 0 <= start_frame < len(events):
            events[start_frame] |= 1
        for bounce in (shot.t_bounce1, shot.t_bounce2, shot.t_bounce3):
            if shot.t_start <= bounce < end:
                frame = int(np.floor(bounce / stride + 0.5))
                if frame < len(events):
                    events[frame] |= 2
    return events


def simulate_rally(
    index: int,
    generation: GenerationConfig,
    camera: CameraSampling,
    physics: GeneratorConfig,
) -> tuple[RallyResult, np.ndarray, tuple[PinholeCamera, ...], int]:
    torch.set_num_threads(1)
    seed = generation.seed + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    attempts = 0
    stride = physics.rally.sim_fps // physics.rally.output_fps
    # Extract event ownership at native physics resolution, before FPS rounding.
    native_rally = replace(physics.rally, output_fps=physics.rally.sim_fps)

    def proposal() -> tuple[RallyResult, np.ndarray, tuple[PinholeCamera, ...]] | None:
        nonlocal attempts
        attempts += 1
        simulator = RallySimulator(
            physics_config=physics.physics.sample(),
            rally_config=native_rally,
            cell_manager=CellManager(),
            targeted_velocity_config=physics.targeted_velocity,
            device="cpu",
        )
        rally = simulator.generate_rally(
            from_cell=int(rng.integers(9)),
            from_side="near" if rng.random() < 0.5 else "far",
        )
        xyz = rally.trajectory[::stride].numpy().astype(np.float32)
        outside = (np.abs(xyz[:, 0]) > HALF_DOUBLES_WIDTH + SIDELINE_CLEAR) | (
            np.abs(xyz[:, 1]) > HALF_LENGTH + BASELINE_CLEAR
        )
        if outside.any():
            xyz = xyz[: int(np.flatnonzero(outside)[0])]
        if len(xyz) < generation.min_frames:
            return None
        cameras = sample_visible_cameras(camera, xyz, rng)
        if cameras is None:
            return None
        return rally, xyz, cameras

    name = f"rally_{index:06d}"
    result, xyz, cameras = generate_with_bounded_physics_resampling(
        proposal, scene_id=name, maximum_attempts=generation.maximum_attempts
    )
    return result, xyz, cameras, attempts
