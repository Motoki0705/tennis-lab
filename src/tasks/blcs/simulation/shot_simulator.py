"""Shot simulator for BLCS (blcs.md §4 compliant).

Generates single shots with:
- Initial condition sampling (position, velocity, spin)
- Physics simulation until termination
- Event detection (bounce, net, fence)
- Shot classification
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.tasks.blcs.simulation.ball_physics import (
    BallPhysics,
    BallState,
    PhysicsConfig,
)
from src.tasks.blcs.simulation.cell_manager import CellManager, ShotCategory

if TYPE_CHECKING:
    pass


@dataclass
class ShotConfig:
    """Configuration for shot generation."""

    # Initial height range (hitting height)
    z_range: tuple[float, float] = (0.8, 1.4)

    # Initial speed range (m/s)
    speed_range: tuple[float, float] = (15.0, 35.0)

    # Azimuth angle range (degrees, 0 = straight ahead)
    azimuth_range_deg: tuple[float, float] = (-30.0, 30.0)

    # Elevation angle range (degrees, positive = upward)
    elevation_range_deg: tuple[float, float] = (5.0, 25.0)

    # Spin ranges (rad/s)
    # Topspin: ω_y < 0 (in our coordinate system)
    spin_x_range: tuple[float, float] = (-20.0, 20.0)
    spin_y_range: tuple[float, float] = (-80.0, -40.0)  # Topspin default
    spin_z_range: tuple[float, float] = (-20.0, 20.0)

    # Simulation parameters
    max_sim_frames: int = 2000  # Max frames at 240 Hz (~8 seconds)
    output_fps: int = 30  # Output frame rate
    sim_fps: int = 240  # Simulation frame rate


@dataclass
class ShotResult:
    """Result of a shot simulation."""

    # Trajectory data (at output_fps)
    trajectory: Tensor  # [T, 3] positions in meters
    velocities: Tensor  # [T, 3] velocities in m/s
    trajectory_sim: Tensor  # [T_sim, 3] full sim resolution (for debugging)

    # Initial state
    initial_state: BallState

    # Event times (frame index at output_fps, -1 if not occurred)
    t_net: int  # Net collision time
    t_fence: int  # Fence collision time
    t_bounce1: int  # First bounce time
    t_bounce2: int  # Second bounce time

    # Event positions
    net_pos: Tensor | None  # Position at net collision
    bounce1_pos: Tensor | None  # First bounce position
    bounce2_pos: Tensor | None  # Second bounce position

    # Classification
    category: ShotCategory
    to_cell: int | None  # Target cell ID (None for DIRECT_NET/DIRECT_FENCE)

    # Metadata
    from_cell: int
    from_side: str
    target_side: str


class ShotSimulator:
    """Generates and simulates single tennis shots.

    Workflow:
    1. Sample initial conditions from specified from_cell
    2. Simulate physics until termination condition
    3. Record events (bounces, net, fence)
    4. Classify shot result
    """

    def __init__(
        self,
        physics_config: PhysicsConfig | None = None,
        shot_config: ShotConfig | None = None,
        cell_manager: CellManager | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize shot simulator.

        Args:
            physics_config: Physics parameters.
            shot_config: Shot generation parameters.
            cell_manager: Cell manager for position sampling.
            device: Torch device.

        """
        self.physics = BallPhysics(physics_config)
        self.shot_config = shot_config or ShotConfig()
        self.cell_manager = cell_manager or CellManager()
        self.device = torch.device(device)

    def sample_initial_condition(
        self,
        from_cell: int,
        from_side: str,
    ) -> BallState:
        """Sample initial conditions for a shot (blcs.md §4).

        Args:
            from_cell: Starting cell ID (0-19).
            from_side: "near" or "far" - which side the shot starts from.

        Returns:
            BallState: Initial ball state.

        """
        cfg = self.shot_config

        # Sample position in cell
        position = self.cell_manager.sample_position_in_cell(
            cell_id=from_cell,
            side=from_side,
            z_range=cfg.z_range,
            device=self.device,
        )

        # Sample velocity (speed + direction)
        velocity = self._sample_velocity(from_side)

        # Sample spin
        spin = self._sample_spin()

        return BallState(position=position, velocity=velocity, spin=spin)

    def _sample_velocity(self, from_side: str) -> Tensor:
        """Sample initial velocity vector.

        Args:
            from_side: "near" or "far".

        Returns:
            Tensor: Velocity [3] in m/s.

        """
        cfg = self.shot_config

        # Sample speed
        speed = cfg.speed_range[0] + torch.rand(1).item() * (
            cfg.speed_range[1] - cfg.speed_range[0]
        )

        # Sample azimuth (horizontal direction)
        azimuth_deg = cfg.azimuth_range_deg[0] + torch.rand(1).item() * (
            cfg.azimuth_range_deg[1] - cfg.azimuth_range_deg[0]
        )
        azimuth_rad = math.radians(azimuth_deg)

        # Sample elevation (vertical angle)
        elevation_deg = cfg.elevation_range_deg[0] + torch.rand(1).item() * (
            cfg.elevation_range_deg[1] - cfg.elevation_range_deg[0]
        )
        elevation_rad = math.radians(elevation_deg)

        # Convert to velocity components
        # Base direction: toward opposite side
        # near side (y < 0) shoots toward +y
        # far side (y > 0) shoots toward -y
        base_dir = 1.0 if from_side == "near" else -1.0

        # Horizontal components
        vx = speed * math.cos(elevation_rad) * math.sin(azimuth_rad)
        vy = speed * math.cos(elevation_rad) * math.cos(azimuth_rad) * base_dir
        vz = speed * math.sin(elevation_rad)

        return torch.tensor([vx, vy, vz], device=self.device)

    def _sample_spin(self) -> Tensor:
        """Sample spin angular velocity.

        Returns:
            Tensor: Spin [3] in rad/s.

        """
        cfg = self.shot_config

        spin_x = cfg.spin_x_range[0] + torch.rand(1).item() * (
            cfg.spin_x_range[1] - cfg.spin_x_range[0]
        )
        spin_y = cfg.spin_y_range[0] + torch.rand(1).item() * (
            cfg.spin_y_range[1] - cfg.spin_y_range[0]
        )
        spin_z = cfg.spin_z_range[0] + torch.rand(1).item() * (
            cfg.spin_z_range[1] - cfg.spin_z_range[0]
        )

        return torch.tensor([spin_x, spin_y, spin_z], device=self.device)

    def simulate_shot(
        self,
        initial_state: BallState,
        from_cell: int,
        from_side: str,
    ) -> ShotResult:
        """Simulate a shot until termination.

        Termination conditions:
        - Second bounce occurs
        - Fence is reached (after first bounce, if no second bounce yet)
        - Max simulation frames reached

        Args:
            initial_state: Initial ball state.
            from_cell: Starting cell ID.
            from_side: "near" or "far".

        Returns:
            ShotResult: Complete shot result.

        """
        cfg = self.shot_config
        target_side = "far" if from_side == "near" else "near"

        # Simulation state
        state = initial_state.clone()
        trajectory_sim: list[Tensor] = [state.position.clone()]
        velocities_sim: list[Tensor] = [state.velocity.clone()]

        # Event tracking
        bounce_count = 0
        t_net_sim = -1
        t_fence_sim = -1
        t_bounce1_sim = -1
        t_bounce2_sim = -1
        net_pos: Tensor | None = None
        bounce1_pos: Tensor | None = None
        bounce2_pos: Tensor | None = None
        hit_net_before_bounce = False
        hit_fence_before_bounce = False

        dt = 1.0 / cfg.sim_fps

        for frame in range(cfg.max_sim_frames - 1):
            prev_pos = state.position.clone()

            # Physics step
            state = self.physics.step(state)

            # Check net collision (before bounce check)
            if bounce_count == 0 and t_net_sim < 0:
                hit_net, pos_at_net = self.physics.check_net_collision(
                    prev_pos, state.position
                )
                if hit_net:
                    t_net_sim = frame + 1
                    net_pos = pos_at_net
                    hit_net_before_bounce = True
                    # Apply net collision (reflect and reduce velocity)
                    state = self.physics.apply_net_collision(state, net_pos=pos_at_net)

            # Check fence collision
            if self.physics.check_fence_collision(state.position):
                if bounce_count == 0:
                    hit_fence_before_bounce = True
                t_fence_sim = frame + 1
                # Record position and stop
                trajectory_sim.append(state.position.clone())
                velocities_sim.append(state.velocity.clone())
                break

            # Handle bounce
            state, bounced = self.physics.handle_bounce(state)
            if bounced:
                bounce_count += 1
                if bounce_count == 1:
                    t_bounce1_sim = frame + 1
                    bounce1_pos = state.position.clone()
                elif bounce_count == 2:
                    t_bounce2_sim = frame + 1
                    bounce2_pos = state.position.clone()
                    # Termination: second bounce
                    trajectory_sim.append(state.position.clone())
                    velocities_sim.append(state.velocity.clone())
                    break

            trajectory_sim.append(state.position.clone())
            velocities_sim.append(state.velocity.clone())

        # Stack trajectories
        trajectory_sim_tensor = torch.stack(trajectory_sim, dim=0)
        velocities_sim_tensor = torch.stack(velocities_sim, dim=0)

        # Downsample to output fps
        downsample_factor = cfg.sim_fps // cfg.output_fps
        trajectory = trajectory_sim_tensor[::downsample_factor]
        velocities = velocities_sim_tensor[::downsample_factor]

        # Convert event times to output fps
        def convert_time(t_sim: int) -> int:
            if t_sim < 0:
                return -1
            return t_sim // downsample_factor

        t_net = convert_time(t_net_sim)
        t_fence = convert_time(t_fence_sim)
        t_bounce1 = convert_time(t_bounce1_sim)
        t_bounce2 = convert_time(t_bounce2_sim)

        # Classify shot
        category, to_cell = self.cell_manager.classify_shot(
            hit_net_before_bounce=hit_net_before_bounce,
            hit_fence_before_bounce=hit_fence_before_bounce,
            bounce_pos=bounce1_pos,
            target_side=target_side,
        )

        return ShotResult(
            trajectory=trajectory,
            velocities=velocities,
            trajectory_sim=trajectory_sim_tensor,
            initial_state=initial_state,
            t_net=t_net,
            t_fence=t_fence,
            t_bounce1=t_bounce1,
            t_bounce2=t_bounce2,
            net_pos=net_pos,
            bounce1_pos=bounce1_pos,
            bounce2_pos=bounce2_pos,
            category=category,
            to_cell=to_cell,
            from_cell=from_cell,
            from_side=from_side,
            target_side=target_side,
        )

    def generate_shot(self, from_cell: int, from_side: str) -> ShotResult:
        """Generate a complete shot from sampling to simulation.

        Args:
            from_cell: Starting cell ID (0-19).
            from_side: "near" or "far".

        Returns:
            ShotResult: Complete shot result.

        """
        initial_state = self.sample_initial_condition(from_cell, from_side)
        return self.simulate_shot(initial_state, from_cell, from_side)
