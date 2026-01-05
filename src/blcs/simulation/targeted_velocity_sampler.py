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

if TYPE_CHECKING:
    from src.blcs.simulation.cell_manager import CellManager


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

    # Minimum/maximum speed constraints (m/s)
    min_speed: float = 12.0
    max_speed: float = 40.0

    # Gravity for projectile calculation (m/s²)
    gravity: float = 9.81


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
    ) -> Tensor:
        """Compute velocity to reach target using projectile approximation.

        Uses simplified projectile motion (gravity only) to estimate
        the required initial velocity. The actual simulation with
        drag/magnus will cause deviation from this ideal trajectory.

        Args:
            start_pos: Starting position [3] (x, y, z).
            target_pos: Target position [3] (typically z=0 for bounce).
            from_side: "near" or "far" - determines base direction.

        Returns:
            Tensor: Velocity [3] in m/s.

        """
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
        # atan2(dx, dy * base_dir) gives angle from base direction
        azimuth_rad = math.atan2(dx, abs(dy))

        # Add noise to azimuth
        azimuth_noise = (
            (torch.rand(1).item() - 0.5) * 2 * math.radians(cfg.azimuth_noise_deg)
        )
        azimuth_rad += azimuth_noise

        # Find elevation and speed using projectile equation
        # For a ball starting at height z0, landing at z=0:
        # Using range equation: R = v² * sin(2θ) / g for flat ground
        # Modified for height difference:
        # t = (v*sin(θ) + sqrt((v*sin(θ))² + 2*g*z0)) / g
        # R = v*cos(θ) * t

        # Try different elevations and find one that works
        best_elevation = math.radians(15.0)  # default
        best_speed = 25.0  # default

        for elev_deg in [8, 12, 15, 18, 22, 26, 30]:
            elev_rad = math.radians(elev_deg)

            # Simplified estimation using range equation with height adjustment
            # R = v² * sin(2θ) / g is for flat ground
            # We adjust for starting height z0

            sin_2elev = math.sin(2 * elev_rad)
            if sin_2elev < 0.1:
                continue

            # Base speed from range equation
            speed_base = math.sqrt(horizontal_dist * cfg.gravity / sin_2elev)

            # Adjust for starting height (ball starts higher, can go further)
            # Approximate correction factor
            height_factor = 1.0 - 0.1 * z0 / max(horizontal_dist, 1.0)
            speed = speed_base * max(0.8, height_factor)

            if cfg.min_speed <= speed <= cfg.max_speed:
                best_elevation = elev_rad
                best_speed = speed
                break

        # Add noise to elevation
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

        return torch.tensor([vx, vy, vz], device=self.device, dtype=torch.float32)

    def sample_velocity_for_target_cell(
        self,
        start_pos: Tensor,
        target_cell: int,
        target_side: str,
        from_side: str,
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

        return self.compute_velocity_to_target(start_pos, target_pos, from_side)
