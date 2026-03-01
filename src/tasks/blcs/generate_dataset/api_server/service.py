"""Simulation orchestration for the WebUI API.

This is the only layer that knows about:
- request schema -> simulator configs
- targeted vs. random velocity sampling
- converting torch tensors to JSON-friendly lists
"""

from __future__ import annotations

import random

import numpy as np
import torch

from src.tasks.blcs.generate_dataset.api_server.metrics import (
    apex_height_m,
    net_clearance_m,
    time_to_bounce1_s,
)
from src.tasks.blcs.generate_dataset.api_server.schemas import (
    ShotEvents,
    ShotLabels,
    ShotMetrics,
    SimulateShotRequest,
    SimulateShotResponse,
    Vec3,
)
from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallState, PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.shot_simulator import ShotConfig, ShotSimulator
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)


def simulate_shot(req: SimulateShotRequest) -> SimulateShotResponse:
    if req.seed is not None:
        random.seed(req.seed)
        np.random.seed(req.seed)
        torch.manual_seed(req.seed)

    # ---- Sim params ----
    # BallPhysics integrates using `physics.dt` while ShotSimulator downsamples based on
    # `shot_config.sim_fps/output_fps`. Keep them consistent by deriving dt from sim_fps,
    # or (if dt is provided) validating that dt and sim_fps agree.
    sim_fps_req = int(req.sim.sim_fps) if req.sim.sim_fps is not None else None
    dt_req = float(req.physics.dt) if req.physics.dt is not None else None

    if dt_req is not None:
        sim_fps_from_dt = int(round(1.0 / dt_req))
        if sim_fps_from_dt <= 0:
            raise ValueError(
                f"Derived sim_fps from dt must be positive, got {sim_fps_from_dt}"
            )
        if sim_fps_req is not None and sim_fps_req != sim_fps_from_dt:
            raise ValueError(
                "Inconsistent simulation timing: physics.dt and sim.sim_fps disagree "
                f"(dt={dt_req} -> sim_fps≈{sim_fps_from_dt}, but sim_fps={sim_fps_req})."
            )
        sim_fps = sim_fps_from_dt
        dt = dt_req
    else:
        sim_fps = sim_fps_req if sim_fps_req is not None else 240
        dt = 1.0 / float(sim_fps)

    output_fps = int(req.sim.output_fps) if req.sim.output_fps is not None else 30
    if sim_fps % output_fps != 0:
        # Redundant with schema validation when both are provided, but also catches defaults.
        raise ValueError(f"sim_fps ({sim_fps}) must be divisible by output_fps ({output_fps})")

    max_sim_frames = (
        int(req.sim.max_sim_frames) if req.sim.max_sim_frames is not None else 2000
    )

    # ---- Physics config ----
    physics = PhysicsConfig(
        gravity=float(req.physics.gravity) if req.physics.gravity is not None else 9.81,
        k_drag=float(req.physics.k_drag) if req.physics.k_drag is not None else 0.01,
        k_magnus=float(req.physics.k_magnus)
        if req.physics.k_magnus is not None
        else 0.001,
        e_z=float(req.physics.e_z) if req.physics.e_z is not None else 0.75,
        mu=float(req.physics.mu) if req.physics.mu is not None else 0.1,
        alpha_net=float(req.physics.alpha_net)
        if req.physics.alpha_net is not None
        else 0.3,
        dt=dt,
        use_drag=bool(req.physics.use_drag) if req.physics.use_drag is not None else True,
        use_magnus=bool(req.physics.use_magnus)
        if req.physics.use_magnus is not None
        else True,
    )

    shot_cfg = ShotConfig(
        # Keep defaults unless we explicitly expose these.
        max_sim_frames=max_sim_frames,
        sim_fps=sim_fps,
        output_fps=output_fps,
    )

    simulator = ShotSimulator(physics_config=physics, shot_config=shot_cfg, device="cpu")

    # ---- Initial state sampling / overrides ----
    # Position:
    if req.shot.position is not None:
        pos = _vec3_to_tensor(req.shot.position)
    else:
        pos = simulator.cell_manager.sample_position_in_cell(
            cell_id=req.from_cell,
            side=req.from_side,
            z_range=shot_cfg.z_range,
            device="cpu",
        )

    # Velocity:
    vel = None
    if req.shot.velocity is not None:
        vel = _vec3_to_tensor(req.shot.velocity)
    else:
        vel = _sample_velocity(req, pos, simulator=simulator)

    # Spin:
    if req.shot.spin is not None:
        spin = _vec3_to_tensor(req.shot.spin)
    else:
        spin = simulator._sample_spin()

    initial_state = BallState(position=pos, velocity=vel, spin=spin)

    # ---- Simulate ----
    result = simulator.simulate_shot(initial_state, req.from_cell, req.from_side)

    # ---- Metrics ----
    apex = apex_height_m(result.trajectory)
    t_b1 = time_to_bounce1_s(result.t_bounce1, fps_out=shot_cfg.output_fps)
    net_clear = net_clearance_m(result.trajectory_sim)

    return SimulateShotResponse(
        positions=result.trajectory.detach().cpu().tolist(),
        velocities=result.velocities.detach().cpu().tolist(),
        fps_out=shot_cfg.output_fps,
        sim_fps=shot_cfg.sim_fps,
        events=ShotEvents(
            t_net=int(result.t_net),
            t_fence=int(result.t_fence),
            t_bounce1=int(result.t_bounce1),
            t_bounce2=int(result.t_bounce2),
            net_pos=_tensor_to_vec3(result.net_pos),
            bounce1_pos=_tensor_to_vec3(result.bounce1_pos),
            bounce2_pos=_tensor_to_vec3(result.bounce2_pos),
        ),
        labels=ShotLabels(
            category=result.category.value,
            to_cell=int(result.to_cell) if result.to_cell is not None else None,
        ),
        metrics=ShotMetrics(
            apex_height_m=float(apex),
            time_to_bounce1_s=t_b1,
            net_clearance_m=net_clear,
        ),
    )


def _sample_velocity(
    req: SimulateShotRequest, pos: torch.Tensor, simulator: ShotSimulator
) -> torch.Tensor:
    # Targeted velocity for `cell` / `point`, otherwise random sampling.
    if req.target_mode == "cell":
        target_side = "far" if req.from_side == "near" else "near"
        sampler = TargetedVelocitySampler(
            cell_manager=simulator.cell_manager,
            config=TargetedVelocityConfig(gravity=simulator.physics.config.gravity),
            device="cpu",
        )
        assert req.to_cell is not None
        return sampler.sample_velocity_for_target_cell(
            start_pos=pos,
            target_cell=req.to_cell,
            target_side=target_side,
            from_side=req.from_side,
        )

    if req.target_mode == "point":
        sampler = TargetedVelocitySampler(
            cell_manager=simulator.cell_manager,
            config=TargetedVelocityConfig(gravity=simulator.physics.config.gravity),
            device="cpu",
        )
        assert req.target_point is not None
        target_pos = torch.tensor(
            [req.target_point.x, req.target_point.y, 0.0],
            dtype=torch.float32,
            device="cpu",
        )
        return sampler.compute_velocity_to_target(pos, target_pos, from_side=req.from_side)

    return simulator._sample_velocity(req.from_side)


def _vec3_to_tensor(v: Vec3) -> torch.Tensor:
    return torch.tensor([v.x, v.y, v.z], dtype=torch.float32, device="cpu")


def _tensor_to_vec3(t: torch.Tensor | None) -> Vec3 | None:
    if t is None:
        return None
    return Vec3(x=float(t[0].item()), y=float(t[1].item()), z=float(t[2].item()))
