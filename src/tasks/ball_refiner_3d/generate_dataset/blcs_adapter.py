"""Translate BLCS physics into refiner rallies with their physics record."""

from __future__ import annotations

import random

import numpy as np
import torch

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
from src.tasks.blcs.generate_dataset.simulation.physics_record import (
    build_physics_record,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyResult,
    RallySimulator,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.physics.ball.record import BallPhysicsRecord
from src.utils.schema.court import (
    BASELINE_CLEAR,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
)


def simulate_rally(
    index: int,
    generation: GenerationConfig,
    camera: CameraSampling,
    physics: GeneratorConfig,
) -> tuple[RallyResult, BallPhysicsRecord, np.ndarray, tuple[PinholeCamera, ...], int]:
    """One accepted rally, truncated at the first fence exit, and its record."""
    torch.set_num_threads(1)
    seed = generation.seed + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    attempts = 0

    def proposal() -> (
        tuple[RallyResult, BallPhysicsRecord, np.ndarray, tuple[PinholeCamera, ...]]
        | None
    ):
        nonlocal attempts
        attempts += 1
        sampled = physics.physics.sample()
        simulator = RallySimulator(
            physics_config=sampled,
            rally_config=physics.rally,
            cell_manager=CellManager(),
            targeted_velocity_config=physics.targeted_velocity,
            device="cpu",
        )
        rally = simulator.generate_rally(
            from_cell=int(rng.integers(9)),
            from_side="near" if rng.random() < 0.5 else "far",
        )
        xyz = rally.trajectory.numpy().astype(np.float32)
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
        return rally, build_physics_record(rally, sampled, len(xyz)), xyz, cameras

    name = f"rally_{index:06d}"
    result, record, xyz, cameras = generate_with_bounded_physics_resampling(
        proposal, scene_id=name, maximum_attempts=generation.maximum_attempts
    )
    return result, record, xyz, cameras, attempts
