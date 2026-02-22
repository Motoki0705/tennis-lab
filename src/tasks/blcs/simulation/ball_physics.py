"""Ball physics simulation for BLCS (blcs.md §2-3 compliant).

Provides realistic tennis ball trajectory generation including:
- Projectile motion with gravity
- Air drag (velocity-squared drag)
- Magnus effect (spin-induced lift)
- Bounce physics with friction
- Net collision with velocity reduction
- Fence collision detection
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    NET_HEIGHT_CENTER,
    NET_HEIGHT_POST,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

if TYPE_CHECKING:
    pass


# Default physics constants (blcs.md §2)
DEFAULT_GRAVITY = 9.81  # m/s^2
DEFAULT_K_DRAG = 0.01  # drag coefficient (simplified)
DEFAULT_K_MAGNUS = 0.001  # magnus coefficient (simplified)
DEFAULT_E_Z = 0.75  # coefficient of restitution (bounce)
DEFAULT_MU = 0.1  # friction coefficient (tangential)
DEFAULT_ALPHA_NET = 0.3  # net velocity reduction factor
DEFAULT_DT = 1 / 240  # simulation time step (240 Hz)


@dataclass
class PhysicsConfig:
    """Configuration for ball physics simulation."""

    gravity: float = DEFAULT_GRAVITY
    k_drag: float = DEFAULT_K_DRAG
    k_magnus: float = DEFAULT_K_MAGNUS
    e_z: float = DEFAULT_E_Z
    mu: float = DEFAULT_MU
    alpha_net: float = DEFAULT_ALPHA_NET
    dt: float = DEFAULT_DT
    use_drag: bool = True
    use_magnus: bool = True


@dataclass
class BallState:
    """State of the ball at a given time."""

    position: Tensor  # [3] (X, Y, Z) in meters
    velocity: Tensor  # [3] (Vx, Vy, Vz) in m/s
    spin: Tensor  # [3] (ωx, ωy, ωz) angular velocity

    def clone(self) -> BallState:
        """Create a deep copy of this state."""
        return BallState(
            position=self.position.clone(),
            velocity=self.velocity.clone(),
            spin=self.spin.clone(),
        )


class CollisionType(Enum):
    """Type of collision event."""

    NONE = "none"
    BOUNCE = "bounce"
    NET = "net"
    FENCE = "fence"


@dataclass
class SimulationEvent:
    """Record of a simulation event."""

    event_type: CollisionType
    time_step: int
    position: Tensor


class BallPhysics:
    """Tennis ball physics simulation (blcs.md §2-3 compliant).

    Implements:
    - Semi-implicit Euler integration (§2.3)
    - Gravity + drag + Magnus force model (§2.2)
    - Bounce with restitution and friction (§3.1)
    - Net collision with velocity reduction (§3.2)
    - Fence boundary detection (§3.3)
    """

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        """Initialize ball physics.

        Args:
            config: Physics configuration. Uses defaults if None.

        """
        self.config = config or PhysicsConfig()

    def compute_acceleration(self, state: BallState) -> Tensor:
        """Compute acceleration given current state (blcs.md §2.2).

        a = a_g + a_d + a_m
          = (0, 0, -g) - k_d ||v|| v + k_m (ω × v)

        Args:
            state: Current ball state.

        Returns:
            Tensor: Acceleration [3].

        """
        device = state.position.device
        cfg = self.config

        # Gravity
        accel = torch.tensor([0.0, 0.0, -cfg.gravity], device=device)

        # Air drag (opposes velocity)
        if cfg.use_drag:
            speed = state.velocity.norm()
            if speed > 1e-6:
                drag_accel = -cfg.k_drag * speed * state.velocity
                accel = accel + drag_accel

        # Magnus effect (spin-induced lift)
        if cfg.use_magnus and state.spin is not None:
            magnus_accel = cfg.k_magnus * torch.cross(state.spin, state.velocity)
            accel = accel + magnus_accel

        return accel

    def step(self, state: BallState) -> BallState:
        """Advance ball state by one time step using semi-implicit Euler (blcs.md §2.3).

        1. a_t = f(p_t, v_t, ω_0)
        2. v_{t+1} = v_t + a_t * dt
        3. p_{t+1} = p_t + v_{t+1} * dt

        Args:
            state: Current ball state.

        Returns:
            BallState: New ball state after dt.

        """
        dt = self.config.dt

        # Compute acceleration
        accel = self.compute_acceleration(state)

        # Semi-implicit Euler: update velocity first, then position
        new_vel = state.velocity + accel * dt
        new_pos = state.position + new_vel * dt

        return BallState(
            position=new_pos,
            velocity=new_vel,
            spin=state.spin.clone(),
        )

    def handle_bounce(self, state: BallState) -> tuple[BallState, bool]:
        """Handle ground bounce if ball is below ground (blcs.md §3.1).

        Bounce detection: Z_t > 0 and Z_{t+1} <= 0 and V_z < 0

        Reflection:
        - V_z' = -e_z * V_z
        - V_x' = (1 - μ) * V_x
        - V_y' = (1 - μ) * V_y
        - Z' = 0

        Args:
            state: Current ball state.

        Returns:
            tuple: (new_state, did_bounce)

        """
        cfg = self.config

        if state.position[2] <= 0 and state.velocity[2] < 0:
            # Bounce occurred
            new_pos = state.position.clone()
            new_pos[2] = 0.0  # Clip to ground

            new_vel = state.velocity.clone()
            new_vel[2] = -cfg.e_z * state.velocity[2]  # Reflect and reduce
            new_vel[0] = (1 - cfg.mu) * state.velocity[0]  # Friction
            new_vel[1] = (1 - cfg.mu) * state.velocity[1]

            return BallState(
                position=new_pos,
                velocity=new_vel,
                spin=state.spin.clone(),
            ), True

        return state, False

    def check_net_collision(
        self,
        prev_pos: Tensor,
        curr_pos: Tensor,
    ) -> tuple[bool, Tensor | None]:
        """Check if ball trajectory crosses the net (blcs.md §3.2).

        Net is at y=0, height varies from NET_HEIGHT_CENTER (center)
        to NET_HEIGHT_POST (edges).

        Args:
            prev_pos: Previous position [3].
            curr_pos: Current position [3].

        Returns:
            tuple: (hit_net, position_at_net)

        """
        # Check if y crosses 0
        if not (
            (prev_pos[1] > 0 and curr_pos[1] <= 0)
            or (prev_pos[1] < 0 and curr_pos[1] >= 0)
        ):
            return False, None

        # Interpolate position at y=0
        t = prev_pos[1] / (prev_pos[1] - curr_pos[1] + 1e-8)
        x_at_net = (prev_pos[0] + t * (curr_pos[0] - prev_pos[0])).item()
        z_at_net = (prev_pos[2] + t * (curr_pos[2] - prev_pos[2])).item()

        # Check if within net width
        if abs(x_at_net) <= HALF_DOUBLES_WIDTH:
            net_height = self._net_height_at_x(x_at_net)

            if z_at_net < net_height:
                net_pos = torch.tensor(
                    [x_at_net, 0.0, z_at_net],
                    device=prev_pos.device,
                )
                return True, net_pos

        return False, None

    def compute_net_clearance(
        self,
        prev_pos: Tensor,
        curr_pos: Tensor,
    ) -> float | None:
        """Compute net clearance when crossing the net plane.

        Args:
            prev_pos: Previous position [3].
            curr_pos: Current position [3].

        Returns:
            float: Clearance in meters (positive = above net).
            None: If net plane was not crossed in this segment.

        """
        if not (
            (prev_pos[1] > 0 and curr_pos[1] <= 0)
            or (prev_pos[1] < 0 and curr_pos[1] >= 0)
        ):
            return None

        t = prev_pos[1] / (prev_pos[1] - curr_pos[1] + 1e-8)
        x_at_net = (prev_pos[0] + t * (curr_pos[0] - prev_pos[0])).item()
        z_at_net = (prev_pos[2] + t * (curr_pos[2] - prev_pos[2])).item()

        if abs(x_at_net) > HALF_DOUBLES_WIDTH:
            return float("inf")

        net_height = self._net_height_at_x(x_at_net)
        return z_at_net - net_height

    def apply_net_collision(
        self,
        state: BallState,
        net_pos: Tensor | None = None,
    ) -> BallState:
        """Apply net collision response (blcs.md §3.2).

        The collision reflects the Y velocity (bounce-back) and reduces
        overall speed by alpha_net.

        Args:
            state: Current ball state.
            net_pos: Optional collision position at the net plane.

        Returns:
            BallState: State with reflected and reduced velocity.

        """
        cfg = self.config
        new_pos = state.position.clone()
        if net_pos is not None:
            new_pos = net_pos.clone()
        else:
            new_pos[1] = 0.0

        new_vel = state.velocity * cfg.alpha_net
        new_vel[1] = -new_vel[1]

        return BallState(
            position=new_pos,
            velocity=new_vel,
            spin=state.spin.clone(),
        )

    def _net_height_at_x(self, x_at_net: float) -> float:
        """Get net height at a given x position (meters)."""
        x_ratio = min(abs(x_at_net) / HALF_DOUBLES_WIDTH, 1.0)
        return NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER)

    def check_fence_collision(self, pos: Tensor) -> bool:
        """Check if ball has reached fence boundary (blcs.md §3.3).

        Fence bounds: [X_MIN, X_MAX] × [Y_MIN, Y_MAX]

        Args:
            pos: Current position [3].

        Returns:
            bool: True if outside fence.

        """
        x, y, z = pos[0].item(), pos[1].item(), pos[2].item()

        # Check if outside fence rectangle
        if x <= X_MIN or x >= X_MAX:
            return True
        if y <= Y_MIN or y >= Y_MAX:
            return True

        return False

    def is_in_singles_court(self, pos: Tensor, target_side: str) -> bool:
        """Check if position is within singles court on target side.

        Args:
            pos: Position [3] (x, y, z).
            target_side: "near" (y < 0) or "far" (y > 0).

        Returns:
            bool: True if in singles court on target side.

        """
        x, y = pos[0].item(), pos[1].item()

        # Check x bounds (singles width)
        if abs(x) > HALF_SINGLES_WIDTH:
            return False

        # Check y bounds based on target side
        if target_side == "far":
            return 0 < y <= HALF_LENGTH
        else:  # near
            return -HALF_LENGTH <= y < 0

    def normalize_position(self, pos: Tensor) -> Tensor:
        """Normalize position to BLCS coordinates (blcs.md §0).

        x_norm = X / HALF_DOUBLES_WIDTH
        y_norm = Y / HALF_LENGTH
        z_norm = Z / NET_HEIGHT_POST

        Args:
            pos: Position [3] or [T, 3] in meters.

        Returns:
            Tensor: Normalized position.

        """
        norm = pos.clone()
        if pos.dim() == 1:
            norm[0] = pos[0] / HALF_DOUBLES_WIDTH
            norm[1] = pos[1] / HALF_LENGTH
            norm[2] = pos[2] / NET_HEIGHT_POST
        else:
            norm[..., 0] = pos[..., 0] / HALF_DOUBLES_WIDTH
            norm[..., 1] = pos[..., 1] / HALF_LENGTH
            norm[..., 2] = pos[..., 2] / NET_HEIGHT_POST
        return norm

    def denormalize_position(self, norm: Tensor) -> Tensor:
        """Denormalize position from BLCS coordinates to meters.

        Args:
            norm: Normalized position [3] or [T, 3].

        Returns:
            Tensor: Position in meters.

        """
        pos = norm.clone()
        if norm.dim() == 1:
            pos[0] = norm[0] * HALF_DOUBLES_WIDTH
            pos[1] = norm[1] * HALF_LENGTH
            pos[2] = norm[2] * NET_HEIGHT_POST
        else:
            pos[..., 0] = norm[..., 0] * HALF_DOUBLES_WIDTH
            pos[..., 1] = norm[..., 1] * HALF_LENGTH
            pos[..., 2] = norm[..., 2] * NET_HEIGHT_POST
        return pos
