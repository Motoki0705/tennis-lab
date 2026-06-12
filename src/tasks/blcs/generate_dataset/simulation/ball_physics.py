"""Ball physics simulation for BLCS.

Provides realistic tennis ball trajectory generation including:
- Projectile motion with gravity (configurable)
- Air drag (velocity-squared drag, relative to wind)
- Magnus effect (spin-induced lift)
- Wind force
- Bounce physics with friction
- Net collision with velocity reduction
- Fence collision detection

All environment constants (gravity, drag, restitution, etc.) can be
perturbed per-scene via ``PhysicsConfig.sample()``.
"""

from __future__ import annotations

import math
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


# Default physics constants
DEFAULT_GRAVITY = 9.81
DEFAULT_K_DRAG = 0.01
DEFAULT_K_MAGNUS = 0.001
DEFAULT_E_Z = 0.75
DEFAULT_MU = 0.1
DEFAULT_ALPHA_NET = 0.3
DEFAULT_ALPHA_NET_CORD = 0.1
DEFAULT_ALPHA_FENCE = 0.3
DEFAULT_NET_HALF_THICKNESS = 0.03
DEFAULT_NET_CORD_RADIUS = 0.03
DEFAULT_DT = 1 / 240


@dataclass
class PhysicsConfig:
    """Configuration for ball physics simulation.

    Includes per-scene perturbation ranges. When ``*_range`` fields are set,
    ``sample()`` draws from them uniformly to create a stochastic config.
    """

    gravity: float = DEFAULT_GRAVITY
    k_drag: float = DEFAULT_K_DRAG
    k_magnus: float = DEFAULT_K_MAGNUS
    e_z: float = DEFAULT_E_Z
    mu: float = DEFAULT_MU
    alpha_net: float = DEFAULT_ALPHA_NET
    alpha_net_cord: float = DEFAULT_ALPHA_NET_CORD
    alpha_fence: float = DEFAULT_ALPHA_FENCE
    net_half_thickness: float = DEFAULT_NET_HALF_THICKNESS
    net_cord_radius: float = DEFAULT_NET_CORD_RADIUS
    dt: float = DEFAULT_DT
    use_drag: bool = True
    use_magnus: bool = True

    # Wind velocity (m/s) in world frame (x, y, z)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # --- Per-scene perturbation ranges (None = use base value) ---
    gravity_range: tuple[float, float] | None = None
    k_drag_range: tuple[float, float] | None = None
    k_magnus_range: tuple[float, float] | None = None
    e_z_range: tuple[float, float] | None = None
    mu_range: tuple[float, float] | None = None
    wind_speed_range: tuple[float, float] | None = None
    wind_direction_range_deg: tuple[float, float] | None = None

    def sample(self) -> PhysicsConfig:
        """Return a new config with stochastic parameters sampled.

        Scalar fields are sampled uniformly from their ``*_range`` if set.
        Wind is sampled from speed and direction ranges if set.
        """

        def _u(base: float, rng: tuple[float, float] | None) -> float:
            if rng is None:
                return base
            lo, hi = rng
            return lo + torch.rand(1).item() * (hi - lo)

        gravity = _u(self.gravity, self.gravity_range)
        k_drag = _u(self.k_drag, self.k_drag_range)
        k_magnus = _u(self.k_magnus, self.k_magnus_range)
        e_z = _u(self.e_z, self.e_z_range)
        mu = _u(self.mu, self.mu_range)

        wind = self.wind
        if self.wind_speed_range is not None:
            speed = _u(0.0, self.wind_speed_range)
            if self.wind_direction_range_deg is not None:
                dir_deg = _u(0.0, self.wind_direction_range_deg)
            else:
                dir_deg = torch.rand(1).item() * 360.0
            dir_rad = math.radians(dir_deg)
            wind = (
                speed * math.cos(dir_rad),
                speed * math.sin(dir_rad),
                0.0,
            )

        return PhysicsConfig(
            gravity=gravity,
            k_drag=k_drag,
            k_magnus=k_magnus,
            e_z=e_z,
            mu=mu,
            alpha_net=self.alpha_net,
            alpha_net_cord=self.alpha_net_cord,
            alpha_fence=self.alpha_fence,
            net_half_thickness=self.net_half_thickness,
            net_cord_radius=self.net_cord_radius,
            dt=self.dt,
            use_drag=self.use_drag,
            use_magnus=self.use_magnus,
            wind=wind,
            # Ranges are NOT propagated to sampled config (it is deterministic)
            gravity_range=None,
            k_drag_range=None,
            k_magnus_range=None,
            e_z_range=None,
            mu_range=None,
            wind_speed_range=None,
            wind_direction_range_deg=None,
        )

    def to_dict(self) -> dict:
        """Serialize to dict (for scene metadata)."""
        return {
            "gravity": self.gravity,
            "k_drag": self.k_drag,
            "k_magnus": self.k_magnus,
            "e_z": self.e_z,
            "mu": self.mu,
            "alpha_net": self.alpha_net,
            "alpha_net_cord": self.alpha_net_cord,
            "alpha_fence": self.alpha_fence,
            "net_half_thickness": self.net_half_thickness,
            "net_cord_radius": self.net_cord_radius,
            "dt": self.dt,
            "use_drag": self.use_drag,
            "use_magnus": self.use_magnus,
            "wind": list(self.wind),
        }


@dataclass
class BallState:
    """State of the ball at a given time."""

    position: Tensor  # [3] (X, Y, Z) in meters
    velocity: Tensor  # [3] (Vx, Vy, Vz) in m/s
    spin: Tensor  # [3] (omega_x, omega_y, omega_z) angular velocity

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
    """Tennis ball physics simulation.

    Implements:
    - Semi-implicit Euler integration
    - Gravity + drag + Magnus force model (drag relative to wind)
    - Bounce with restitution and friction
    - Net collision with velocity reduction
    - Fence boundary detection
    """

    def __init__(self, config: PhysicsConfig | None = None) -> None:
        """Initialize ball physics.

        Args:
            config: Physics configuration. Uses defaults if None.
        """
        self.config = config or PhysicsConfig()
        self._wind_vec: Tensor | None = None

    @property
    def wind_vec(self) -> Tensor:
        """Lazily materialized wind vector."""
        if self._wind_vec is None:
            self._wind_vec = torch.tensor(self.config.wind, dtype=torch.float32)
        return self._wind_vec

    def compute_acceleration(self, state: BallState) -> Tensor:
        """Compute acceleration given current state.

        a = a_g + a_d + a_m
        Drag and Magnus use velocity relative to wind.

        Args:
            state: Current ball state.

        Returns:
            Acceleration [3].
        """
        device = state.position.device
        cfg = self.config

        # Gravity
        accel = torch.tensor([0.0, 0.0, -cfg.gravity], device=device)

        # Wind-relative velocity
        wind = self.wind_vec.to(device)
        v_rel = state.velocity - wind

        # Air drag (opposes relative velocity)
        if cfg.use_drag:
            speed_rel = v_rel.norm()
            if speed_rel > 1e-6:
                drag_accel = -cfg.k_drag * speed_rel * v_rel
                accel = accel + drag_accel

        # Magnus effect (spin-induced lift, relative velocity)
        if cfg.use_magnus and state.spin is not None:
            magnus_accel = cfg.k_magnus * torch.linalg.cross(state.spin, v_rel)
            accel = accel + magnus_accel

        return accel

    def step(self, state: BallState) -> BallState:
        """Advance ball state by one time step using semi-implicit Euler.

        1. a_t = f(p_t, v_t, omega_0)
        2. v_{t+1} = v_t + a_t * dt
        3. p_{t+1} = p_t + v_{t+1} * dt

        Args:
            state: Current ball state.

        Returns:
            New ball state after dt.
        """
        dt = self.config.dt

        accel = self.compute_acceleration(state)

        new_vel = state.velocity + accel * dt
        new_pos = state.position + new_vel * dt

        return BallState(
            position=new_pos,
            velocity=new_vel,
            spin=state.spin.clone(),
        )

    def handle_bounce(self, state: BallState) -> tuple[BallState, bool]:
        """Handle ground bounce if ball is below ground.

        Reflection:
        - V_z' = -e_z * V_z
        - V_x' = (1 - mu) * V_x
        - V_y' = (1 - mu) * V_y
        - Z' = 0

        Args:
            state: Current ball state.

        Returns:
            (new_state, did_bounce)
        """
        cfg = self.config

        if state.position[2] <= 0 and state.velocity[2] < 0:
            new_pos = state.position.clone()
            new_pos[2] = 0.0

            new_vel = state.velocity.clone()
            new_vel[2] = -cfg.e_z * state.velocity[2]
            new_vel[0] = (1 - cfg.mu) * state.velocity[0]
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
        """Check collision with explicit net geometry.

        Net profile (Y-Z cross-section) is modeled as:
        - Mesh body: |y| <= net_half_thickness and z <= (net_height - net_cord_radius)
        - Cord cap: upper half-ellipse above mesh body (kamaboko profile)

        Args:
            prev_pos: Previous position [3].
            curr_pos: Current position [3].

        Returns:
            (hit_net, position_at_net)
        """
        half_t = max(float(self.config.net_half_thickness), 1e-4)
        if curr_pos[1].item() > half_t and prev_pos[1].item() > half_t:
            return False, None
        if curr_pos[1].item() < -half_t and prev_pos[1].item() < -half_t:
            return False, None

        prev_y = prev_pos[1].item()
        curr_y = curr_pos[1].item()
        dy = curr_y - prev_y

        # Use first contact with near-side face of net slab.
        if abs(dy) < 1e-8:
            t = 0.0
        elif prev_y <= -half_t < curr_y:
            t = (-half_t - prev_y) / dy
        elif prev_y >= half_t > curr_y:
            t = (half_t - prev_y) / dy
        else:
            t = prev_y / (prev_y - curr_y + 1e-8)

        if t < 0.0 or t > 1.0:
            return False, None

        net_pos = prev_pos + t * (curr_pos - prev_pos)
        x_at_net = net_pos[0].item()
        y_at_net = net_pos[1].item()
        z_at_net = net_pos[2].item()

        if abs(x_at_net) > HALF_DOUBLES_WIDTH:
            return False, None

        top_z = self._net_profile_top_z_at_y(
            x_at_net=x_at_net,
            y_at_net=y_at_net,
        )
        if top_z is None:
            return False, None
        if z_at_net <= top_z:
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
            Clearance in metres (positive = above net), or None if not crossed.
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
        """Apply net collision response.

        Uses explicit net geometry normal:
        - Mesh region: normal along +/-Y
        - Cord region: normal from cap gradient

        Args:
            state: Current ball state.
            net_pos: Optional collision position at the net plane.

        Returns:
            State with reflected and reduced velocity.
        """
        cfg = self.config
        new_pos = state.position.clone()
        if net_pos is not None:
            new_pos = net_pos.clone()
        else:
            new_pos[1] = 0.0

        normal, is_cord_region = self._net_surface_normal(
            net_pos=new_pos,
            incoming_velocity=state.velocity,
        )
        alpha = cfg.alpha_net_cord if is_cord_region else cfg.alpha_net

        new_vel = state.velocity * alpha
        v_normal = torch.dot(new_vel, normal)
        if v_normal < 0:
            new_vel = new_vel - 2.0 * v_normal * normal

        return BallState(
            position=new_pos,
            velocity=new_vel,
            spin=state.spin.clone(),
        )

    def _net_profile_top_z_at_y(
        self,
        x_at_net: float,
        y_at_net: float,
    ) -> float | None:
        """Return top z of the explicit net profile at (x, y)."""
        half_t = max(float(self.config.net_half_thickness), 1e-4)
        if abs(y_at_net) > half_t:
            return None

        cord_r = max(float(self.config.net_cord_radius), 1e-4)
        net_height = self._net_height_at_x(x_at_net)
        mesh_top = net_height - cord_r
        yy = y_at_net / half_t
        cap_term = max(0.0, 1.0 - yy * yy)
        cap_top = mesh_top + cord_r * math.sqrt(cap_term)
        return cap_top

    def _net_surface_normal(
        self,
        net_pos: Tensor,
        incoming_velocity: Tensor,
    ) -> tuple[Tensor, bool]:
        """Compute net surface normal at collision point.

        Returns:
            (normal, is_cord_region)
        """
        x_at_net = float(net_pos[0].item())
        y_at_net = float(net_pos[1].item())
        z_at_net = float(net_pos[2].item())

        half_t = max(float(self.config.net_half_thickness), 1e-4)
        cord_r = max(float(self.config.net_cord_radius), 1e-4)
        net_height = self._net_height_at_x(x_at_net)
        mesh_top = net_height - cord_r

        vy = float(incoming_velocity[1].item())
        y_fallback = -1.0 if vy >= 0.0 else 1.0

        device = incoming_velocity.device
        dtype = incoming_velocity.dtype

        if z_at_net <= mesh_top:
            normal = torch.tensor([0.0, y_fallback, 0.0], device=device, dtype=dtype)
            return normal, False

        # Cord cap (upper half-ellipse): F(y,z)= (y/half_t)^2 + ((z-mesh_top)/cord_r)^2 - 1
        dFy = 2.0 * y_at_net / (half_t * half_t)
        dFz = 2.0 * (z_at_net - mesh_top) / (cord_r * cord_r)

        ny = dFy
        nz = dFz
        if abs(ny) < 1e-8 and abs(nz) < 1e-8:
            ny = y_fallback
            nz = 0.0
        elif abs(ny) < 1e-8:
            ny = 0.15 * y_fallback

        normal = torch.tensor([0.0, ny, nz], device=device, dtype=dtype)
        normal = normal / (normal.norm() + 1e-8)
        return normal, True

    def _net_height_at_x(self, x_at_net: float) -> float:
        """Get net height at a given x position (metres)."""
        x_ratio = min(abs(x_at_net) / HALF_DOUBLES_WIDTH, 1.0)
        return NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER)

    def check_fence_collision(
        self,
        prev_pos: Tensor,
        curr_pos: Tensor,
    ) -> tuple[bool, Tensor | None, Tensor | None]:
        """Check if segment crosses a fence plane.

        Args:
            prev_pos: Previous position [3].
            curr_pos: Current position [3].

        Returns:
            (hit_fence, position_at_fence, inward_normal).
        """
        prev_x = prev_pos[0].item()
        prev_y = prev_pos[1].item()
        curr_x = curr_pos[0].item()
        curr_y = curr_pos[1].item()

        prev_inside = X_MIN <= prev_x <= X_MAX and Y_MIN <= prev_y <= Y_MAX
        curr_inside = X_MIN <= curr_x <= X_MAX and Y_MIN <= curr_y <= Y_MAX
        if curr_inside or not prev_inside:
            return False, None, None

        plane_candidates: list[tuple[float, float, Tensor]] = []
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        device = prev_pos.device
        dtype = prev_pos.dtype

        if curr_x < X_MIN and abs(dx) > 1e-8:
            t = (X_MIN - prev_x) / dx
            if 0.0 <= t <= 1.0:
                plane_candidates.append(
                    (t, X_MIN, torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype))
                )
        if curr_x > X_MAX and abs(dx) > 1e-8:
            t = (X_MAX - prev_x) / dx
            if 0.0 <= t <= 1.0:
                plane_candidates.append(
                    (t, X_MAX, torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype))
                )
        if curr_y < Y_MIN and abs(dy) > 1e-8:
            t = (Y_MIN - prev_y) / dy
            if 0.0 <= t <= 1.0:
                plane_candidates.append(
                    (t, Y_MIN, torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype))
                )
        if curr_y > Y_MAX and abs(dy) > 1e-8:
            t = (Y_MAX - prev_y) / dy
            if 0.0 <= t <= 1.0:
                plane_candidates.append(
                    (t, Y_MAX, torch.tensor([0.0, -1.0, 0.0], device=device, dtype=dtype))
                )

        if not plane_candidates:
            return False, None, None

        t_hit, _, normal = min(plane_candidates, key=lambda x: x[0])
        fence_pos = prev_pos + t_hit * (curr_pos - prev_pos)

        return True, fence_pos, normal

    def apply_fence_collision(
        self,
        state: BallState,
        fence_pos: Tensor,
        fence_normal: Tensor,
    ) -> BallState:
        """Apply fence collision response.

        Reflects normal velocity component and reduces speed by alpha_fence.
        """
        cfg = self.config
        new_pos = fence_pos.clone() + fence_normal * 1e-4

        new_vel = state.velocity * cfg.alpha_fence
        v_normal = torch.dot(new_vel, fence_normal)
        if v_normal < 0:
            new_vel = new_vel - 2.0 * v_normal * fence_normal

        return BallState(
            position=new_pos,
            velocity=new_vel,
            spin=state.spin.clone(),
        )

    def is_in_singles_court(self, pos: Tensor, target_side: str) -> bool:
        """Check if position is within singles court on target side.

        Args:
            pos: Position [3] (x, y, z).
            target_side: ``"near"`` (y < 0) or ``"far"`` (y > 0).

        Returns:
            True if in singles court on target side.
        """
        x, y = pos[0].item(), pos[1].item()

        if abs(x) > HALF_SINGLES_WIDTH:
            return False

        if target_side == "far":
            return 0 < y <= HALF_LENGTH
        else:
            return -HALF_LENGTH <= y < 0

    def normalize_position(self, pos: Tensor) -> Tensor:
        """Normalize position to BLCS coordinates.

        x_norm = X / HALF_DOUBLES_WIDTH
        y_norm = Y / HALF_LENGTH
        z_norm = Z / NET_HEIGHT_POST

        Args:
            pos: Position [3] or [T, 3] in metres.

        Returns:
            Normalized position.
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
        """Denormalize position from BLCS coordinates to metres.

        Args:
            norm: Normalized position [3] or [T, 3].

        Returns:
            Position in metres.
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
