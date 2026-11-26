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
from src.tasks.blcs.generate_dataset.simulation.net_contact import (
    NetContact,
    NetContactPlan,
    NetContactProposalRejected,
)
from src.tasks.blcs.generate_dataset.simulation.physics_record import (
    build_physics_record,
)
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyEndReason,
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
    net_height_at_x,
)


def simulate_rally(
    index: int,
    generation: GenerationConfig,
    camera: CameraSampling,
    physics: GeneratorConfig,
    net_contact: NetContact | None = None,
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
        plan = None
        if net_contact is NetContact.CROSSED:
            plan = NetContactPlan(
                int(rng.integers(physics.rally.max_rallies - 1)), net_contact
            )
        elif net_contact is NetContact.DROPPED:
            plan = NetContactPlan(
                int(rng.integers(1, physics.rally.max_rallies)), net_contact
            )
        try:
            rally = simulator.generate_rally(
                from_cell=int(rng.integers(9)),
                from_side="near" if rng.random() < 0.5 else "far",
                net_plan=plan,
            )
        except NetContactProposalRejected:
            return None
        xyz = rally.trajectory.numpy().astype(np.float32)
        outside = (np.abs(xyz[:, 0]) > HALF_DOUBLES_WIDTH + SIDELINE_CLEAR) | (
            np.abs(xyz[:, 1]) > HALF_LENGTH + BASELINE_CLEAR
        )
        if outside.any():
            xyz = xyz[: int(np.flatnonzero(outside)[0])]
        if len(xyz) < generation.min_frames:
            return None
        if generation.net_contact_fraction > 0 and not _matches_net_request(
            rally, frames=len(xyz), cord_radius=sampled.net_cord_radius, expected=plan
        ):
            return None
        cameras = sample_visible_cameras(camera, xyz, rng)
        if cameras is None:
            return None
        record = build_physics_record(
            rally, sampled, len(xyz), verify_bits=generation.net_contact_fraction > 0
        )
        return rally, record, xyz, cameras

    name = f"rally_{index:06d}"
    result, record, xyz, cameras = generate_with_bounded_physics_resampling(
        proposal, scene_id=name, maximum_attempts=generation.maximum_attempts
    )
    return result, record, xyz, cameras, attempts


def _matches_net_request(
    rally: RallyResult,
    *,
    frames: int,
    cord_radius: float,
    expected: NetContactPlan | None,
) -> bool:
    stride = rally.sim_fps // rally.fps_out
    contacts = [
        event
        for event in rally.physics_events
        if event.kind == "net" and event.step < frames * stride
    ]
    if expected is None:
        return not contacts
    if len(contacts) != 1 or contacts[0].step > (frames - 1) * stride:
        return False
    net = contacts[0]
    events = [
        event for event in rally.physics_events if event.step <= (frames - 1) * stride
    ]
    shots = [event for event in events if event.kind == "shot"]
    if sum(shot.step < net.step for shot in shots) != expected.shot_index + 1:
        return False
    bounce = next(
        (event for event in events if event.kind == "bounce" and event.step > net.step),
        None,
    )
    if bounce is None:
        return False
    signed_depth = bounce.position[1].item() * net.velocity_before[1].item()
    if expected.outcome is NetContact.DROPPED:
        return (
            signed_depth < 0
            and not any(shot.step > net.step for shot in shots)
            and rally.end_reason is RallyEndReason.NET_FAULT
        )
    x, _, z = net.position.tolist()
    return (
        z > net_height_at_x(x) - cord_radius
        and signed_depth > 0
        and any(shot.step > bounce.step for shot in shots)
    )
