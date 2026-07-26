"""Targeted velocity sampler for cell-to-cell shots.

Computes an initial velocity that aims at a target cell. The speed is first
derived analytically from gravity-only projectile motion. When a physics
simulator is provided, the landing point is then refined iteratively against
the full physics model (drag/Magnus/wind): the shot is simulated to its first
bounce, the landing error is measured, and a virtual aim point is shifted by
that error (shooting method). Net hits raise the elevation and retry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.utils.schema.court import HALF_DOUBLES_WIDTH, net_height_at_x

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallPhysics
    from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager


@dataclass
class TargetedVelocityConfig:
    """Configuration for targeted velocity sampling."""

    # Elevation ranges for profile sampling (degrees)
    drive_elevation_range_deg: tuple[float, float] = (5.0, 25.0)
    lob_elevation_range_deg: tuple[float, float] = (35.0, 70.0)
    lob_probability: float = 0.0

    # Maximum gravity-only apex for any targeted shot. This prevents a long
    # target combined with a high lob angle from creating implausible 20m+
    # trajectories while preserving the requested landing point.
    max_ballistic_apex_height_m: float = 8.0

    # Gravity for projectile calculation (m/s²)
    gravity: float = 9.81

    # Net-hit resampling
    net_retry_max_attempts: int = 12
    net_check_max_frames: int = 600
    net_elevation_step_deg: float = 2.0

    # Physics-based landing refinement (shooting method). Enabled whenever a
    # physics simulator is passed to the sampler. Each iteration simulates the
    # shot to its first bounce and shifts a virtual aim point by the landing
    # error so drag/Magnus/wind are compensated.
    landing_refine_enabled: bool = True
    landing_refine_max_iters: int = 14
    landing_refine_tolerance_m: float = 0.25
    landing_sim_max_frames: int = 1200

    # Margin (metres) from cell edges when sampling target bounce positions.
    # Keeps refined landings clear of court boundaries despite residual error.
    target_margin_m: float = 0.35


@dataclass(frozen=True)
class _NetCheckResult:
    """Result of the short net-only physics check."""

    passed: bool
    hit_pos: Tensor | None = None


@dataclass(frozen=True)
class _LandingResult:
    """Result of simulating a shot until its first bounce.

    ``bounce_pos`` is the first bounce position, or a fence-contact proxy when
    the ball reaches the fence before bouncing. ``None`` if the ball hit the
    net or never landed within the frame budget.
    """

    bounce_pos: Tensor | None
    hit_net: bool
    net_pos: Tensor | None


class TargetedVelocitySampler:
    """Samples velocity to aim at specific target cells.

    Uses projectile motion approximation (ignoring drag/magnus) to compute a
    velocity toward a target cell. It does not refine against the final landing
    point; it only resamples elevation when a short physics check detects a net
    hit.
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
        from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager

        self.cell_manager = cell_manager or CellManager()
        self.config = config or TargetedVelocityConfig()
        self.device = torch.device(device)
        if self.config.max_ballistic_apex_height_m <= 0.0:
            raise ValueError("max_ballistic_apex_height_m must be positive.")

    def compute_velocity_to_target(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_deg: float | None = None,
        profile: str | None = None,
        physics: BallPhysics | None = None,
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
            physics: Optional physics simulator for net-hit checking.
            spin: Optional spin vector for net-hit checking (rad/s).

        Returns:
            Tensor: Velocity [3] in m/s.

        """
        elevation_rad = self._sample_elevation_rad(elevation_deg, profile)

        if physics is not None and self.config.landing_refine_enabled:
            return self._compute_velocity_with_landing_refinement(
                start_pos=start_pos,
                target_pos=target_pos,
                from_side=from_side,
                elevation_rad=elevation_rad,
                physics=physics,
                spin=spin,
            )

        return self._compute_velocity_with_net_retry(
            start_pos=start_pos,
            target_pos=target_pos,
            from_side=from_side,
            elevation_rad=elevation_rad,
            physics=physics,
            spin=spin,
        )

    def _compute_velocity_with_net_retry(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_rad: float,
        physics: BallPhysics | None,
        spin: Tensor | None,
    ) -> Tensor:
        """Gravity-only aiming with net-hit elevation retry (legacy path)."""
        horizontal_dist = float(
            torch.linalg.norm(target_pos[:2] - start_pos[:2]).item()
        )
        elevation_step_rad = math.radians(self.config.net_elevation_step_deg)
        max_attempts = max(1, self.config.net_retry_max_attempts)
        last_velocity: Tensor | None = None

        for attempt in range(max_attempts):
            velocity = self._compute_velocity_to_target_once(
                start_pos=start_pos,
                target_pos=target_pos,
                from_side=from_side,
                elevation_rad=elevation_rad,
            )
            last_velocity = velocity

            net_check = self._check_net_passage(
                start_pos=start_pos,
                velocity=velocity,
                physics=physics,
                spin=spin,
            )
            if net_check.passed:
                return velocity

            if attempt < max_attempts - 1:
                elevation_rad = self._raise_elevation_after_net_hit(
                    elevation_rad=elevation_rad,
                    hit_pos=net_check.hit_pos,
                    start_pos=start_pos,
                    horizontal_dist=horizontal_dist,
                    default_step_rad=elevation_step_rad,
                )

        assert last_velocity is not None
        return last_velocity

    def _compute_velocity_with_landing_refinement(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_rad: float,
        physics: BallPhysics,
        spin: Tensor | None,
    ) -> Tensor:
        """Refine aim against full physics so the first bounce hits the target.

        Shooting method: simulate the candidate shot to its first bounce,
        measure the landing error in the ground plane, and shift a virtual aim
        point by that error before recomputing the gravity-only solution. Net
        hits raise the elevation instead (reusing the deficit-based step).
        """
        cfg = self.config
        elevation_step_rad = math.radians(cfg.net_elevation_step_deg)
        target_side = "far" if from_side == "near" else "near"

        virtual_target = target_pos.clone()
        best_velocity: Tensor | None = None
        best_error = float("inf")
        best_virtual: Tensor | None = None
        best_error_vec: Tensor | None = None
        prev_error: float | None = None
        # Step gain; halved whenever an update makes the landing error worse
        # (strong Magnus can make the full step overshoot and oscillate).
        gain = 1.0

        for _ in range(max(1, cfg.landing_refine_max_iters)):
            velocity = self._compute_velocity_to_target_once(
                start_pos=start_pos,
                target_pos=virtual_target,
                from_side=from_side,
                elevation_rad=elevation_rad,
            )
            landing = self._simulate_landing(
                start_pos=start_pos,
                velocity=velocity,
                spin=spin,
                physics=physics,
            )

            if landing.hit_net:
                horizontal_dist = float(
                    torch.linalg.norm(virtual_target[:2] - start_pos[:2]).item()
                )
                elevation_rad = self._raise_elevation_after_net_hit(
                    elevation_rad=elevation_rad,
                    hit_pos=landing.net_pos,
                    start_pos=start_pos,
                    horizontal_dist=horizontal_dist,
                    default_step_rad=elevation_step_rad,
                )
                continue

            if landing.bounce_pos is None:
                # Never landed within budget; nothing to correct against.
                elevation_rad += elevation_step_rad
                continue

            error_vec = landing.bounce_pos[:2] - target_pos[:2]
            error = float(torch.linalg.norm(error_vec).item())

            if error < best_error:
                best_error = error
                best_velocity = velocity
                best_virtual = virtual_target.clone()
                best_error_vec = error_vec.clone()
            if error <= cfg.landing_refine_tolerance_m:
                return velocity

            if prev_error is not None and error > prev_error:
                gain = max(0.25, gain * 0.5)
            prev_error = error

            # Step from the best-known aim point with the (possibly damped)
            # gain rather than chaining steps from a diverging iterate.
            assert best_virtual is not None and best_error_vec is not None
            virtual_target = best_virtual.clone()
            virtual_target[:2] = virtual_target[:2] - gain * best_error_vec
            virtual_target = self._clamp_virtual_target(virtual_target, target_side)

        if best_velocity is not None:
            return best_velocity

        # All attempts hit the net or never landed: fall back to legacy path.
        return self._compute_velocity_with_net_retry(
            start_pos=start_pos,
            target_pos=target_pos,
            from_side=from_side,
            elevation_rad=elevation_rad,
            physics=physics,
            spin=spin,
        )

    def _clamp_virtual_target(self, virtual_target: Tensor, target_side: str) -> Tensor:
        """Keep the virtual aim point on the target side and bounded.

        The virtual aim point is allowed well outside the physical court:
        with strong drag a deep shot may require aiming far beyond the fence
        for the gravity-only solution to carry far enough.
        """
        from src.utils.schema.court import X_MAX, Y_MAX

        x_limit = float(X_MAX) * 2.0
        y_limit = float(abs(Y_MAX)) * 2.0
        min_depth = 0.5  # metres beyond the net

        x = float(virtual_target[0].item())
        y = float(virtual_target[1].item())

        x = max(-x_limit, min(x_limit, x))
        if target_side == "far":
            y = max(min_depth, min(y_limit, y))
        else:
            y = min(-min_depth, max(-y_limit, y))

        clamped = virtual_target.clone()
        clamped[0] = x
        clamped[1] = y
        return clamped

    def _simulate_landing(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        spin: Tensor | None,
        physics: BallPhysics,
    ) -> _LandingResult:
        """Simulate a shot with full physics until its first bounce.

        Fence contact before the bounce is treated as a proxy landing so the
        refinement can still pull a long overshoot back toward the target.
        """
        from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallState

        spin_vec = spin if spin is not None else torch.zeros_like(velocity)
        state = BallState(
            position=start_pos.clone(),
            velocity=velocity.clone(),
            spin=spin_vec.clone(),
        )

        for _ in range(self.config.landing_sim_max_frames):
            prev_pos = state.position.clone()
            state = physics.step(state)

            hit_net, net_pos = physics.check_net_collision(prev_pos, state.position)
            if hit_net:
                return _LandingResult(bounce_pos=None, hit_net=True, net_pos=net_pos)

            hit_fence, fence_pos, _ = physics.check_fence_collision(
                prev_pos, state.position
            )
            if hit_fence and fence_pos is not None:
                return _LandingResult(
                    bounce_pos=fence_pos.clone(), hit_net=False, net_pos=None
                )

            state, bounced = physics.handle_bounce(state)
            if bounced:
                return _LandingResult(
                    bounce_pos=state.position.clone(), hit_net=False, net_pos=None
                )

        return _LandingResult(bounce_pos=None, hit_net=False, net_pos=None)

    def _compute_velocity_to_target_once(
        self,
        start_pos: Tensor,
        target_pos: Tensor,
        from_side: str,
        elevation_rad: float,
    ) -> Tensor:
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

        speed = self._solve_speed_for_target(horizontal_dist, z0, elevation_rad)

        # Convert to velocity components
        # vx: horizontal component in x direction
        # vy: horizontal component in y direction (toward opponent)
        # vz: vertical component (upward)
        cos_elev = math.cos(elevation_rad)
        sin_elev = math.sin(elevation_rad)

        vx = speed * cos_elev * math.sin(azimuth_rad)
        vy = speed * cos_elev * math.cos(azimuth_rad) * base_dir
        vz = speed * sin_elev

        ballistic_apex = z0 + max(vz, 0.0) ** 2 / (2.0 * self.config.gravity)
        if ballistic_apex > self.config.max_ballistic_apex_height_m:
            vx, vy, vz = self._velocity_with_capped_ballistic_apex(
                dx=dx,
                dy=dy,
                horizontal_dist=horizontal_dist,
                z0=z0,
            )

        return torch.tensor([vx, vy, vz], device=self.device, dtype=torch.float32)

    def _velocity_with_capped_ballistic_apex(
        self,
        *,
        dx: float,
        dy: float,
        horizontal_dist: float,
        z0: float,
    ) -> tuple[float, float, float]:
        """Solve a target-reaching launch at the configured apex ceiling."""
        gravity = self.config.gravity
        apex = max(z0, self.config.max_ballistic_apex_height_m)
        vz = math.sqrt(max(0.0, 2.0 * gravity * (apex - z0)))
        flight_time = (vz + math.sqrt(vz**2 + 2.0 * gravity * max(z0, 0.0))) / gravity
        if flight_time <= 0.0 or horizontal_dist <= 1e-8:
            return 0.0, 0.0, vz
        horizontal_speed = horizontal_dist / flight_time
        return (
            horizontal_speed * dx / horizontal_dist,
            horizontal_speed * dy / horizontal_dist,
            vz,
        )

    def sample_velocity_for_target_cell(
        self,
        start_pos: Tensor,
        target_cell: int,
        target_side: str,
        from_side: str,
        profile: str | None = None,
        physics: BallPhysics | None = None,
        spin: Tensor | None = None,
    ) -> Tensor:
        """Sample velocity aimed at a specific target cell.

        Samples a random position within the target cell and computes
        velocity to reach that position.

        Args:
            start_pos: Starting position [3].
            target_cell: Target cell ID (0-8).
            target_side: Side of target cell ("near" or "far").
            from_side: Side the shot is coming from.

        Returns:
            Tensor: Velocity [3] in m/s.

        """
        self._validate_cell_id(target_cell)

        # Sample target position within cell (ground level)
        target_pos = self.cell_manager.sample_bounce_position_in_cell(
            cell_id=target_cell,
            side=target_side,
            device=self.device,
            margin=self.config.target_margin_m,
        )

        return self.compute_velocity_to_target(
            start_pos=start_pos,
            target_pos=target_pos,
            from_side=from_side,
            profile=profile,
            physics=physics,
            spin=spin,
        )

    def sample_velocity_for_serve(
        self,
        start_pos: Tensor,
        target_cell: int,
        target_side: str,
        from_side: str,
        physics: BallPhysics | None = None,
        spin: Tensor | None = None,
    ) -> Tensor:
        """Sample velocity aimed at a service box cell.

        Uses lower elevation and higher speed typical of serves.

        Args:
            start_pos: Starting position [3].
            target_cell: Target service box cell ID (0 or 1).
            target_side: Side of target cell.
            from_side: Side the serve is coming from.
            physics: Optional physics for clearance check.
            spin: Optional spin vector.

        Returns:
            Velocity [3] in m/s.
        """
        self._validate_cell_id(target_cell)

        target_pos = self.cell_manager.sample_bounce_position_in_cell(
            cell_id=target_cell,
            side=target_side,
            device=self.device,
            margin=self.config.target_margin_m,
        )

        return self.compute_velocity_to_target(
            start_pos=start_pos,
            target_pos=target_pos,
            from_side=from_side,
            profile="drive",  # Serves use drive (low elevation)
            physics=physics,
            spin=spin,
        )

    def _validate_cell_id(self, cell_id: int) -> None:
        from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
            NUM_CELLS_PER_SIDE,
        )

        if not 0 <= cell_id < NUM_CELLS_PER_SIDE:
            raise ValueError(
                f"target_cell must be in [0, {NUM_CELLS_PER_SIDE - 1}], got {cell_id}"
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

        return math.radians(elev)

    def _solve_speed_for_target(
        self,
        horizontal_dist: float,
        z0: float,
        elevation_rad: float,
    ) -> float:
        cfg = self.config
        if horizontal_dist <= 1e-8:
            return 0.0

        cos_elev = math.cos(elevation_rad)
        tan_elev = math.tan(elevation_rad)
        denominator = 2.0 * cos_elev**2 * (z0 + horizontal_dist * tan_elev)
        if denominator <= 0.0:
            return 0.0
        return math.sqrt(cfg.gravity * horizontal_dist**2 / denominator)

    def _check_net_passage(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        physics: BallPhysics | None,
        spin: Tensor | None,
    ) -> _NetCheckResult:
        if physics is None:
            return self._check_net_passage_gravity(start_pos, velocity)
        return self._check_net_passage_with_physics(
            start_pos=start_pos,
            velocity=velocity,
            spin=spin,
            physics=physics,
        )

    def _check_net_passage_gravity(
        self,
        start_pos: Tensor,
        velocity: Tensor,
    ) -> _NetCheckResult:
        vy = float(velocity[1].item())
        if abs(vy) < 1e-6:
            return _NetCheckResult(passed=False)

        t = -float(start_pos[1].item()) / vy
        if t <= 0.0:
            return _NetCheckResult(passed=False)

        x_at_net = float(start_pos[0].item()) + float(velocity[0].item()) * t
        z_at_net = (
            float(start_pos[2].item())
            + float(velocity[2].item()) * t
            - 0.5 * self.config.gravity * t**2
        )
        hit_pos = torch.tensor(
            [x_at_net, 0.0, z_at_net], device=self.device, dtype=torch.float32
        )
        if abs(x_at_net) > HALF_DOUBLES_WIDTH:
            return _NetCheckResult(passed=True, hit_pos=hit_pos)
        net_height = self._net_height_at_x(x_at_net)
        return _NetCheckResult(passed=z_at_net > net_height, hit_pos=hit_pos)

    def _check_net_passage_with_physics(
        self,
        start_pos: Tensor,
        velocity: Tensor,
        spin: Tensor | None,
        physics: BallPhysics,
    ) -> _NetCheckResult:
        from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallState

        spin_vec = spin if spin is not None else torch.zeros_like(velocity)
        state = BallState(
            position=start_pos.clone(),
            velocity=velocity.clone(),
            spin=spin_vec.clone(),
        )

        for _ in range(self.config.net_check_max_frames):
            prev_pos = state.position.clone()
            state = physics.step(state)

            hit_net, net_pos = physics.check_net_collision(prev_pos, state.position)
            if hit_net:
                return _NetCheckResult(passed=False, hit_pos=net_pos)

            clearance = physics.compute_net_clearance(prev_pos, state.position)
            if clearance is not None:
                return _NetCheckResult(passed=clearance > 0.0)

            state, bounced = physics.handle_bounce(state)
            if bounced:
                return _NetCheckResult(passed=False)

        return _NetCheckResult(passed=False)

    def _raise_elevation_after_net_hit(
        self,
        elevation_rad: float,
        hit_pos: Tensor | None,
        start_pos: Tensor,
        horizontal_dist: float,
        default_step_rad: float,
    ) -> float:
        if hit_pos is None or horizontal_dist <= 1e-8:
            return elevation_rad + default_step_rad

        x_at_net = float(hit_pos[0].item())
        z_at_net = float(hit_pos[2].item())
        deficit = self._net_height_at_x(x_at_net) - z_at_net
        if deficit <= 0.0:
            return elevation_rad + default_step_rad

        dist_to_net = float(torch.linalg.norm(hit_pos[:2] - start_pos[:2]).item())
        scale_dist = max(dist_to_net, horizontal_dist * 0.25, 1.0)
        deficit_step = math.atan(deficit / scale_dist)
        return elevation_rad + max(default_step_rad, deficit_step)

    def _net_height_at_x(self, x_at_net: float) -> float:
        return float(net_height_at_x(x_at_net))
