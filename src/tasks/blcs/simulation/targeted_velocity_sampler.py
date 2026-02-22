"""Targeted velocity sampler for cell-to-cell shots.

Computes initial velocity to aim at specific target cells while
adding realistic variation. Uses simplified projectile motion
(gravity only) for speed, with drag/magnus effects adding natural noise.

The calculation is approximate because:
1. We use gravity-only projectile equations for speed estimation
2. The actual simulation includes drag and magnus forces
3. This discrepancy creates natural variation in landing positions

Additional noise parameters are applied to azimuth, elevation, and speed
to further increase variation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.utils.schema.court import HALF_DOUBLES_WIDTH, NET_HEIGHT_CENTER, NET_HEIGHT_POST

if TYPE_CHECKING:
    from src.blcs.simulation.cell_manager import CellManager
    from src.blcs.simulation.ball_physics import BallPhysics


@dataclass
class TargetedVelocityConfig:
    """Configuration for targeted velocity sampling."""

    # Azimuth noise (degrees, added to computed azimuth)
    azimuth_noise_deg: float = 5.0

    # Elevation noise (degrees, added to computed elevation)
    elevation_noise_deg: float = 3.0

    # Speed variation factor (multiplier range: 1 ± speed_variation)
    speed_variation: float = 0.15

    # Minimum/maximum elevation constraints (degrees)
    min_elevation_deg: float = 3.0
    max_elevation_deg: float = 35.0

    # Elevation ranges for profile sampling (degrees)
    drive_elevation_range_deg: tuple[float, float] = (5.0, 25.0)
    lob_elevation_range_deg: tuple[float, float] = (35.0, 70.0)
    lob_probability: float = 0.0

    # Minimum/maximum speed constraints (m/s)
    min_speed: float = 12.0
    max_speed: float = 40.0

    # Gravity for projectile calculation (m/s²)
    gravity: float = 9.81

    # Speed solve configuration (gravity-only approximation)
    speed_solve_max_iters: int = 32
    speed_solve_tol: float = 1e-3

    # Optional refinement using drag/magnus simulation
    refine_enabled: bool = False
    refine_iters: int = 1
    refine_speed_scale_min: float = 0.7
    refine_speed_scale_max: float = 1.4
    refine_max_azimuth_adjust_deg: float = 15.0
    refine_max_frames: int = 1200

    # Net clearance constraint (optional)
    net_clearance_enabled: bool = False
    net_clearance_min: float = 0.1
    net_clearance_max_attempts: int = 12
    net_clearance_max_frames: int = 600


class TargetedVelocitySampler:
    """Samples velocity to aim at specific target cells.

    Uses projectile motion approximation (ignoring drag/magnus) to
    compute initial velocity that lands ball near target, then adds
    realistic variation.

    The drag and magnus forces in the actual physics simulation will
    cause the ball to deviate from the idealized trajectory, creating
    natural variation in landing positions.
    """

    def __init__(
        self,
        cell_manager: CellManager | None = None,
        config: TargetedVelocityConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize targeted velocity sampler.

        Args:
            cell_manager: Cell manager for court grid operations.
            config: Configuration parameters.
            device: Torch device.

        """
        # Import here to avoid circular dependency
        from src.blcs.simulation.cell_manager import CellManager

        self.cell_manager = cell_manager or CellManager()
        self.config = config or TargetedVelocityConfig()
        self.device = torch.device(device)

    def compute_velocity_to_target(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_deg: float | None = None,
        profile: str | None = None,
        physics: "BallPhysics | None" = None,
        spin: Tensor | None = None,
    ) -> Tensor:
        """Compute velocity to reach target using projectile approximation.

        Uses simplified projectile motion (gravity only) to estimate
        the required initial velocity. The actual simulation with
        drag/magnus will cause deviation from this ideal trajectory.

        Args:
            start_pos: Starting position [3] (x, y, z).
            target_pos: Target position [3] (typically z=0 for bounce).
            from_side: "near" or "far" - determines base direction.
            elevation_deg: Optional elevation angle override (degrees).
            profile: Optional elevation profile ("drive" or "lob").
            physics: Optional physics simulator for refinement.
            spin: Optional spin vector for refinement (rad/s).

        Returns:
            Tensor: Velocity [3] in m/s.

        """
        cfg = self.config

        max_attempts = 1
        if cfg.net_clearance_enabled:
            max_attempts = max(1, cfg.net_clearance_max_attempts)

        last_velocity: Tensor | None = None

        for _ in range(max_attempts):
            velocity = self._compute_velocity_to_target_once(
                start_pos=start_pos,
                target_pos=target_pos,
                from_side=from_side,
                elevation_deg=elevation_deg,
                profile=profile,
                physics=physics,
                spin=spin,
            )
            last_velocity = velocity

            if not cfg.net_clearance_enabled:
                return velocity

            if self._passes_net_clearance(
                start_pos=start_pos,
                velocity=velocity,
                physics=physics,
                spin=spin,
            ):
                return velocity

        if last_velocity is None:
            last_velocity = self._compute_velocity_to_target_once(
                start_pos=start_pos,
                target_pos=target_pos,
                from_side=from_side,
                elevation_deg=elevation_deg,
                profile=profile,
                physics=physics,
                spin=spin,
            )

        return last_velocity

    def _compute_velocity_to_target_once(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_deg: float | None = None,
        profile: str | None = None,
        physics: "BallPhysics | None" = None,
        spin: Tensor | None = None,
    ) -> Tensor:
        cfg = self.config

        # Horizontal displacement
        dx = (target_pos[0] - start_pos[0]).item()
        dy = (target_pos[1] - start_pos[1]).item()
        horizontal_dist = math.sqrt(dx**2 + dy**2)

        # Vertical displacement (start height to ground)
        z0 = start_pos[2].item()

        # Base direction (determines sign of vy)
        base_dir = 1.0 if from_side == "near" else -1.0

        # Compute azimuth (angle in x-y plane from y-axis)
        azimuth_rad = math.atan2(dx, dy * base_dir if dy != 0 else abs(dy) + 1e-6)

        # Add noise to azimuth
        azimuth_noise = (
            (torch.rand(1).item() - 0.5) * 2 * math.radians(cfg.azimuth_noise_deg)
        )
        azimuth_rad += azimuth_noise

        best_elevation = self._sample_elevation_rad(elevation_deg, profile)
        best_speed = self._solve_speed_for_target(
            horizontal_dist, z0, best_elevation
        )

        if best_speed is None:
            best_elevation, best_speed = self._fallback_speed_and_elevation(
                horizontal_dist, z0
            )

        # Add noise to elevation before final clamp
        elevation_noise = (
            (torch.rand(1).item() - 0.5) * 2 * math.radians(cfg.elevation_noise_deg)
        )
        best_elevation = max(
            math.radians(cfg.min_elevation_deg),
            min(math.radians(cfg.max_elevation_deg), best_elevation + elevation_noise),
        )

        # Add noise to speed
        speed_factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * cfg.speed_variation
        best_speed = max(
            cfg.min_speed, min(cfg.max_speed, best_speed * speed_factor)
        )

        # Convert to velocity components
        # vx: horizontal component in x direction
        # vy: horizontal component in y direction (toward opponent)
        # vz: vertical component (upward)
        cos_elev = math.cos(best_elevation)
        sin_elev = math.sin(best_elevation)

        vx = best_speed * cos_elev * math.sin(azimuth_rad)
        vy = best_speed * cos_elev * math.cos(azimuth_rad) * base_dir
        vz = best_speed * sin_elev

        velocity = torch.tensor([vx, vy, vz], device=self.device, dtype=torch.float32)

        if cfg.refine_enabled and physics is not None:
            velocity = self._refine_velocity_to_target(
                start_pos=start_pos,
                target_pos=target_pos,
                velocity=velocity,
                spin=spin,
                physics=physics,
            )

        return velocity

    def sample_velocity_for_target_cell(
        self,
        start_pos: Tensor,
        target_cell: int,
        target_side: str,
        from_side: str,
        profile: str | None = None,
        physics: "BallPhysics | None" = None,
        spin: Tensor | None = None,
    ) -> Tensor:
        """Sample velocity aimed at a specific target cell.

        Samples a random position within the target cell and computes
        velocity to reach that position.

        Args:
            start_pos: Starting position [3].
            target_cell: Target cell ID (0-19).
            target_side: Side of target cell ("near" or "far").
            from_side: Side the shot is coming from.

        Returns:
            Tensor: Velocity [3] in m/s.

        """
        # Sample target position within cell (ground level)
        target_pos = self.cell_manager.sample_bounce_position_in_cell(
            cell_id=target_cell,
            side=target_side,
            device=self.device,
        )

        return self.compute_velocity_to_target(
            start_pos=start_pos,
            target_pos=target_pos,
            from_side=from_side,
            profile=profile,
            physics=physics,
            spin=spin,
        )

    def _sample_elevation_rad(
        self,
        elevation_deg: float | None,
        profile: str | None,
    ) -> float:
        cfg = self.config

        if elevation_deg is None:
            if profile is None:
                use_lob = torch.rand(1).item() < cfg.lob_probability
                profile = "lob" if use_lob else "drive"

            if profile == "lob":
                elev_min, elev_max = cfg.lob_elevation_range_deg
            else:
                elev_min, elev_max = cfg.drive_elevation_range_deg

            elev = elev_min + torch.rand(1).item() * (elev_max - elev_min)
        else:
            elev = elevation_deg

        elev = max(cfg.min_elevation_deg, min(cfg.max_elevation_deg, elev))
        return math.radians(elev)

    def _solve_speed_for_target(
        self,
        horizontal_dist: float,
        z0: float,
        elevation_rad: float,
    ) -> float | None:
        cfg = self.config

        def range_for_speed(speed: float) -> float:
            sin_elev = math.sin(elevation_rad)
            cos_elev = math.cos(elevation_rad)
            v_sin = speed * sin_elev
            t = (v_sin + math.sqrt(max(0.0, v_sin**2 + 2 * cfg.gravity * z0))) / cfg.gravity
            return speed * cos_elev * t

        r_min = range_for_speed(cfg.min_speed)
        r_max = range_for_speed(cfg.max_speed)
        lower = min(r_min, r_max)
        upper = max(r_min, r_max)
        if not (lower <= horizontal_dist <= upper):
            return None

        lo = cfg.min_speed
        hi = cfg.max_speed
        for _ in range(cfg.speed_solve_max_iters):
            mid = 0.5 * (lo + hi)
            r_mid = range_for_speed(mid)
            if abs(r_mid - horizontal_dist) <= cfg.speed_solve_tol:
                return mid
            if r_mid < horizontal_dist:
                lo = mid
            else:
                hi = mid

        return 0.5 * (lo + hi)

    def _fallback_speed_and_elevation(
        self,
        horizontal_dist: float,
        z0: float,
    ) -> tuple[float, float]:
        cfg = self.config
        best_elevation = math.radians(15.0)  # default
        best_speed = 25.0  # default

        for elev_deg in [8, 12, 15, 18, 22, 26, 30]:
            elev_rad = math.radians(elev_deg)
            sin_2elev = math.sin(2 * elev_rad)
            if sin_2elev < 0.1:
                continue

            speed_base = math.sqrt(horizontal_dist * cfg.gravity / sin_2elev)
            height_factor = 1.0 - 0.1 * z0 / max(horizontal_dist, 1.0)
            speed = speed_base * max(0.8, height_factor)

            if cfg.min_speed <= speed <= cfg.max_speed:
                best_elevation = elev_rad
                best_speed = speed
                break

        return best_elevation, best_speed

    def _refine_velocity_to_target(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        velocity: Tensor,
        spin: Tensor | None,
        physics: "BallPhysics",
    ) -> Tensor:
        cfg = self.config
        spin_vec = spin if spin is not None else torch.zeros_like(velocity)
        refined = velocity.clone()

        for _ in range(cfg.refine_iters):
            landing = self._simulate_to_first_bounce(
                start_pos=start_pos,
                velocity=refined,
                spin=spin_vec,
                physics=physics,
            )
            if landing is None:
                break

            desired = target_pos[:2] - start_pos[:2]
            current = landing[:2] - start_pos[:2]
            desired_dist = float(torch.linalg.norm(desired).item())
            current_dist = float(torch.linalg.norm(current).item())
            if current_dist < 1e-6 or desired_dist < 1e-6:
                break

            scale = desired_dist / current_dist
            scale = max(cfg.refine_speed_scale_min, min(cfg.refine_speed_scale_max, scale))

            desired_angle = math.atan2(desired[0].item(), desired[1].item())
            current_angle = math.atan2(current[0].item(), current[1].item())
            delta = desired_angle - current_angle
            max_delta = math.radians(cfg.refine_max_azimuth_adjust_deg)
            delta = max(-max_delta, min(max_delta, delta))

            vx, vy, vz = refined.tolist()
            horiz_speed = math.hypot(vx, vy) * scale
            angle = math.atan2(vx, vy) + delta

            vx = horiz_speed * math.sin(angle)
            vy = horiz_speed * math.cos(angle)

            total_speed = math.sqrt(vx**2 + vy**2 + vz**2)
            if total_speed > cfg.max_speed:
                factor = cfg.max_speed / max(total_speed, 1e-6)
                vx *= factor
                vy *= factor
                vz *= factor
            elif total_speed < cfg.min_speed:
                factor = cfg.min_speed / max(total_speed, 1e-6)
                vx *= factor
                vy *= factor
                vz *= factor

            refined = torch.tensor(
                [vx, vy, vz], device=refined.device, dtype=refined.dtype
            )

        return refined

    def _passes_net_clearance(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        physics: "BallPhysics | None",
        spin: Tensor | None,
    ) -> bool:
        clearance = self._estimate_net_clearance(
            start_pos=start_pos,
            velocity=velocity,
            physics=physics,
            spin=spin,
        )
        if clearance is None:
            return False
        return clearance >= self.config.net_clearance_min

    def _estimate_net_clearance(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        physics: "BallPhysics | None",
        spin: Tensor | None,
    ) -> float | None:
        if physics is None:
            return self._estimate_net_clearance_gravity(start_pos, velocity)
        return self._estimate_net_clearance_with_physics(
            start_pos=start_pos,
            velocity=velocity,
            spin=spin,
            physics=physics,
        )

    def _estimate_net_clearance_gravity(
        self,
        start_pos: Tensor,
        velocity: Tensor,
    ) -> float | None:
        cfg = self.config
        vy = float(velocity[1].item())
        if abs(vy) < 1e-6:
            return None

        t = -float(start_pos[1].item()) / vy
        if t <= 0:
            return None

        x_at_net = float(start_pos[0].item()) + float(velocity[0].item()) * t
        if abs(x_at_net) > HALF_DOUBLES_WIDTH:
            return float("inf")

        z_at_net = (
            float(start_pos[2].item())
            + float(velocity[2].item()) * t
            - 0.5 * cfg.gravity * t**2
        )
        net_height = self._net_height_at_x(x_at_net)
        return z_at_net - net_height

    def _estimate_net_clearance_with_physics(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        spin: Tensor | None,
        physics: "BallPhysics",
    ) -> float | None:
        from src.blcs.simulation.ball_physics import BallState

        spin_vec = spin if spin is not None else torch.zeros_like(velocity)
        state = BallState(
            position=start_pos.clone(),
            velocity=velocity.clone(),
            spin=spin_vec.clone(),
        )

        for _ in range(self.config.net_clearance_max_frames):
            prev_pos = state.position.clone()
            state = physics.step(state)

            clearance = physics.compute_net_clearance(prev_pos, state.position)
            if clearance is not None:
                return clearance

            state, bounced = physics.handle_bounce(state)
            if bounced:
                return None

        return None

    def _net_height_at_x(self, x_at_net: float) -> float:
        x_ratio = min(abs(x_at_net) / HALF_DOUBLES_WIDTH, 1.0)
        return NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER)

    def _simulate_to_first_bounce(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        spin: Tensor,
        physics: "BallPhysics",
    ) -> Tensor | None:
        from src.blcs.simulation.ball_physics import BallState

        state = BallState(
            position=start_pos.clone(),
            velocity=velocity.clone(),
            spin=spin.clone(),
        )

        for _ in range(self.config.refine_max_frames):
            prev_pos = state.position.clone()
            state = physics.step(state)

            hit_net, _ = physics.check_net_collision(prev_pos, state.position)
            if hit_net:
                return None

            state, bounced = physics.handle_bounce(state)
            if bounced:
                return state.position.clone()

        return None
