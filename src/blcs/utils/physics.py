"""Ball physics simulation for BLCS.

Provides realistic tennis ball trajectory generation including:
- Projectile motion with gravity
- Air drag (optional)
- Bounce physics
- Spin effects (simplified)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.blcs.utils.constants import (
    AIR_DENSITY,
    BALL_MASS,
    BALL_RADIUS,
    COR_COURT,
    DRAG_COEFFICIENT,
    GRAVITY,
    NORM_SCALE_X,
    NORM_SCALE_Y,
    NORM_SCALE_Z,
)
from src.utils.geometry import HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_CENTER

if TYPE_CHECKING:
    pass


class ShotType(Enum):
    """Types of tennis shots."""

    FLAT = "flat"
    TOPSPIN = "topspin"
    SLICE = "slice"
    LOB = "lob"
    DROP = "drop"


@dataclass
class BallState:
    """State of the ball at a given time."""

    position: Tensor  # [3] (x, y, z) in meters
    velocity: Tensor  # [3] (vx, vy, vz) in m/s
    spin: Tensor | None = None  # [3] angular velocity (optional)


class BallPhysics:
    """Tennis ball physics simulation.

    Simulates ball trajectories with configurable physics parameters.
    """

    def __init__(
        self,
        gravity: float = GRAVITY,
        drag_coefficient: float = DRAG_COEFFICIENT,
        air_density: float = AIR_DENSITY,
        ball_mass: float = BALL_MASS,
        ball_radius: float = BALL_RADIUS,
        cor: float = COR_COURT,
        use_drag: bool = True,
        use_magnus: bool = False,
    ) -> None:
        """Initialize ball physics.

        Args:
            gravity: Gravitational acceleration (m/s^2).
            drag_coefficient: Aerodynamic drag coefficient.
            air_density: Air density (kg/m^3).
            ball_mass: Ball mass (kg).
            ball_radius: Ball radius (m).
            cor: Coefficient of restitution for bounces.
            use_drag: Whether to apply air drag.
            use_magnus: Whether to apply Magnus effect (spin).

        """
        self.gravity = gravity
        self.drag_coeff = drag_coefficient
        self.air_density = air_density
        self.ball_mass = ball_mass
        self.ball_radius = ball_radius
        self.cor = cor
        self.use_drag = use_drag
        self.use_magnus = use_magnus

        # Precompute drag constant: 0.5 * Cd * rho * A / m
        cross_section = math.pi * ball_radius**2
        self.drag_factor = (
            0.5 * drag_coefficient * air_density * cross_section / ball_mass
        )

    def compute_acceleration(
        self,
        velocity: Tensor,
        spin: Tensor | None = None,
    ) -> Tensor:
        """Compute acceleration given current velocity and spin.

        Args:
            velocity: Current velocity [3].
            spin: Angular velocity [3] (optional).

        Returns:
            Tensor: Acceleration [3].

        """
        # Gravity (always downward in z)
        accel = torch.tensor([0.0, 0.0, -self.gravity], device=velocity.device)

        # Air drag (opposes velocity)
        if self.use_drag:
            speed = velocity.norm()
            if speed > 1e-6:
                drag_accel = -self.drag_factor * speed * velocity
                accel = accel + drag_accel

        # Magnus effect (spin-induced lift)
        if self.use_magnus and spin is not None:
            # Simplified Magnus: F = Cl * (w x v)
            magnus_coeff = 0.1  # simplified coefficient
            magnus_accel = magnus_coeff * torch.cross(spin, velocity)
            accel = accel + magnus_accel

        return accel

    def step(
        self,
        state: BallState,
        dt: float,
    ) -> BallState:
        """Advance the ball state by one time step using RK4.

        Args:
            state: Current ball state.
            dt: Time step in seconds.

        Returns:
            BallState: New ball state after dt.

        """
        pos = state.position
        vel = state.velocity
        spin = state.spin

        # RK4 integration
        k1_v = self.compute_acceleration(vel, spin)
        k1_p = vel

        k2_v = self.compute_acceleration(vel + 0.5 * dt * k1_v, spin)
        k2_p = vel + 0.5 * dt * k1_v

        k3_v = self.compute_acceleration(vel + 0.5 * dt * k2_v, spin)
        k3_p = vel + 0.5 * dt * k2_v

        k4_v = self.compute_acceleration(vel + dt * k3_v, spin)
        k4_p = vel + dt * k3_v

        new_vel = vel + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        new_pos = pos + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

        return BallState(position=new_pos, velocity=new_vel, spin=spin)

    def handle_bounce(
        self,
        state: BallState,
        ground_z: float = 0.0,
    ) -> tuple[BallState, bool]:
        """Handle ground bounce if ball is below ground.

        Args:
            state: Current ball state.
            ground_z: Ground level (default 0).

        Returns:
            tuple: (new_state, did_bounce)

        """
        if state.position[2] < ground_z and state.velocity[2] < 0:
            # Bounce: reflect z velocity with energy loss
            new_pos = state.position.clone()
            new_pos[2] = ground_z + (ground_z - state.position[2])

            new_vel = state.velocity.clone()
            new_vel[2] = -self.cor * state.velocity[2]
            # Reduce horizontal velocity slightly on bounce
            new_vel[0] = 0.9 * state.velocity[0]
            new_vel[1] = 0.9 * state.velocity[1]

            return BallState(position=new_pos, velocity=new_vel, spin=state.spin), True

        return state, False

    def check_net_collision(
        self,
        prev_pos: Tensor,
        curr_pos: Tensor,
    ) -> bool:
        """Check if ball trajectory crosses the net.

        Args:
            prev_pos: Previous position [3].
            curr_pos: Current position [3].

        Returns:
            bool: True if net collision detected.

        """
        # Net is at y=0, from x=-HALF_DOUBLES_WIDTH to x=+HALF_DOUBLES_WIDTH
        # Height varies from NET_HEIGHT_CENTER at center to NET_HEIGHT_POST at edges

        # Check if y crosses 0
        if (prev_pos[1] > 0 and curr_pos[1] < 0) or (
            prev_pos[1] < 0 and curr_pos[1] > 0
        ):
            # Interpolate position at y=0
            t = prev_pos[1] / (prev_pos[1] - curr_pos[1] + 1e-8)
            x_at_net = prev_pos[0] + t * (curr_pos[0] - prev_pos[0])
            z_at_net = prev_pos[2] + t * (curr_pos[2] - prev_pos[2])

            # Check if within net width
            if abs(x_at_net) <= HALF_DOUBLES_WIDTH:
                # Get net height at this x position
                net_height = self._get_net_height(x_at_net.item())
                if z_at_net < net_height:
                    return True

        return False

    def _get_net_height(self, x: float) -> float:
        """Get net height at given x position.

        Net sags in the middle (0.914m) and is higher at posts (1.07m).
        """
        from src.utils.geometry import NET_HEIGHT_POST

        # Linear interpolation from center to edge
        x_ratio = abs(x) / HALF_DOUBLES_WIDTH
        return NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER)

    def simulate_trajectory(
        self,
        initial_pos: Tensor,
        initial_vel: Tensor,
        dt: float = 1 / 30,
        num_frames: int = 60,
        spin: Tensor | None = None,
        stop_on_second_bounce: bool = True,
    ) -> Tensor:
        """Simulate ball trajectory over multiple frames.

        Args:
            initial_pos: Initial position [3] in meters.
            initial_vel: Initial velocity [3] in m/s.
            dt: Time step in seconds.
            num_frames: Number of frames to simulate.
            spin: Angular velocity [3] (optional).
            stop_on_second_bounce: Stop trajectory after second bounce.

        Returns:
            Tensor: Trajectory positions [T, 3] in meters.

        """
        device = initial_pos.device
        trajectory = [initial_pos.clone()]

        state = BallState(
            position=initial_pos.clone(),
            velocity=initial_vel.clone(),
            spin=spin,
        )

        bounce_count = 0

        for _ in range(num_frames - 1):
            prev_pos = state.position.clone()

            # Physics step
            state = self.step(state, dt)

            # Handle bounce
            state, bounced = self.handle_bounce(state)
            if bounced:
                bounce_count += 1
                if stop_on_second_bounce and bounce_count >= 2:
                    # Pad remaining frames with last position
                    remaining = num_frames - len(trajectory)
                    last_pos = trajectory[-1]
                    for _ in range(remaining):
                        trajectory.append(last_pos.clone())
                    break

            # Check net collision (stop if hit)
            if self.check_net_collision(prev_pos, state.position):
                # Ball hit net - stop trajectory
                remaining = num_frames - len(trajectory)
                last_pos = prev_pos
                for _ in range(remaining):
                    trajectory.append(last_pos.clone())
                break

            trajectory.append(state.position.clone())

        return torch.stack(trajectory, dim=0)

    def normalize_trajectory(self, trajectory: Tensor) -> Tensor:
        """Normalize trajectory to BLCS coordinate system.

        Args:
            trajectory: Trajectory [T, 3] in meters.

        Returns:
            Tensor: Normalized trajectory [T, 3].

        """
        normalized = trajectory.clone()
        normalized[:, 0] = trajectory[:, 0] / NORM_SCALE_X
        normalized[:, 1] = trajectory[:, 1] / NORM_SCALE_Y
        normalized[:, 2] = trajectory[:, 2] / NORM_SCALE_Z
        return normalized

    def denormalize_trajectory(self, normalized: Tensor) -> Tensor:
        """Denormalize trajectory from BLCS coordinates to meters.

        Args:
            normalized: Normalized trajectory [T, 3].

        Returns:
            Tensor: Trajectory [T, 3] in meters.

        """
        trajectory = normalized.clone()
        trajectory[:, 0] = normalized[:, 0] * NORM_SCALE_X
        trajectory[:, 1] = normalized[:, 1] * NORM_SCALE_Y
        trajectory[:, 2] = normalized[:, 2] * NORM_SCALE_Z
        return trajectory


def generate_random_shot(
    shot_type: ShotType = ShotType.FLAT,
    from_near_side: bool = True,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Generate random initial conditions for a tennis shot.

    Args:
        shot_type: Type of shot to generate.
        from_near_side: If True, shot starts from near baseline (y < 0).
        device: Device for tensors.

    Returns:
        tuple: (initial_pos [3], initial_vel [3])

    """
    # Starting position (near baseline or far baseline)
    if from_near_side:
        y_start = -HALF_LENGTH + torch.rand(1).item() * 2  # Near baseline area
        y_target = HALF_LENGTH - torch.rand(1).item() * 6  # Target far side
    else:
        y_start = HALF_LENGTH - torch.rand(1).item() * 2  # Far baseline area
        y_target = -HALF_LENGTH + torch.rand(1).item() * 6  # Target near side

    x_start = (torch.rand(1).item() - 0.5) * HALF_DOUBLES_WIDTH * 1.5
    z_start = 0.8 + torch.rand(1).item() * 1.5  # Hit height 0.8-2.3m

    x_target = (torch.rand(1).item() - 0.5) * HALF_DOUBLES_WIDTH * 1.5

    # Calculate required velocity based on shot type
    if shot_type == ShotType.FLAT:
        speed = 20 + torch.rand(1).item() * 15  # 20-35 m/s
        z_target = 0.5 + torch.rand(1).item() * 1.0  # Target height
    elif shot_type == ShotType.TOPSPIN:
        speed = 18 + torch.rand(1).item() * 12  # 18-30 m/s
        z_target = 1.0 + torch.rand(1).item() * 2.0  # Higher arc
    elif shot_type == ShotType.SLICE:
        speed = 15 + torch.rand(1).item() * 10  # 15-25 m/s
        z_target = 0.5 + torch.rand(1).item() * 0.5  # Lower trajectory
    elif shot_type == ShotType.LOB:
        speed = 12 + torch.rand(1).item() * 8  # 12-20 m/s
        z_target = 4.0 + torch.rand(1).item() * 3.0  # High arc
    else:  # DROP
        speed = 8 + torch.rand(1).item() * 6  # 8-14 m/s
        z_target = 1.5 + torch.rand(1).item() * 1.0

    # Direction vector
    dx = x_target - x_start
    dy = y_target - y_start
    dz = z_target - z_start

    # Normalize and scale by speed
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    vx = speed * dx / dist
    vy = speed * dy / dist
    vz = speed * dz / dist

    # Add some upward component for gravity compensation
    vz += 2.0 + torch.rand(1).item() * 3.0

    initial_pos = torch.tensor([x_start, y_start, z_start], device=device)
    initial_vel = torch.tensor([vx, vy, vz], device=device)

    return initial_pos, initial_vel
