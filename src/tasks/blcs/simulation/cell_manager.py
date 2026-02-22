"""Cell management for BLCS shot distribution control (blcs.md §6 extended).

Manages 20-cell grid for both from_cell and to_cell:
- Cells 0-8: Singles court interior (3×3 grid)
- Cells 9-19: Exterior regions (fence area)

Also handles shot category classification:
- DIRECT_NET: Ball hits net before first bounce
- DIRECT_FENCE: Ball reaches fence before first bounce
- IN_COURT: First bounce in singles court (cells 0-8)
- OUT_COURT: First bounce outside court (cells 9-19)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.utils.schema.court import (
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    X_MAX,
    X_MIN,
    Y_MAX,
)

if TYPE_CHECKING:
    pass


class ShotCategory(Enum):
    """Shot outcome category."""

    DIRECT_NET = "direct_net"  # Hit net before first bounce
    DIRECT_FENCE = "direct_fence"  # Reached fence before first bounce
    IN_COURT = "in_court"  # First bounce in singles court
    OUT_COURT = "out_court"  # First bounce outside singles court


@dataclass
class CellBounds:
    """Bounds of a cell."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class CellManager:
    """Manages 20-cell grid for shot distribution control.

    Cell layout (viewed from above, for one side):

    Fence boundary
    ┌────┬────┬────┬────┬────┐
    │ 17 │ 18 │ 19 │ 20 │ 21 │  ← Baseline exterior (mapped to 15-19)
    ├────┼────┼────┼────┼────┤
    │  9 │  6 │  7 │  8 │ 12 │  ← Row 2 (baseline side)
    ├────┼────┼────┼────┼────┤
    │ 10 │  3 │  4 │  5 │ 13 │  ← Row 1 (middle)
    ├────┼────┼────┼────┼────┤
    │ 11 │  0 │  1 │  2 │ 14 │  ← Row 0 (net side)
    └────┴────┴────┴────┴────┘
              Net (y=0)

    Final ID mapping:
    - 0-8: Court interior (3×3)
    - 9-11: Left exterior (3 rows)
    - 12-14: Right exterior (3 rows)
    - 15-19: Baseline exterior (5 columns)
    """

    # Court interior grid boundaries (singles court)
    # X: [-HALF_SINGLES_WIDTH, +HALF_SINGLES_WIDTH] divided into 3
    # Y: [0, HALF_LENGTH] (far side) or [-HALF_LENGTH, 0] (near side) divided into 3

    def __init__(self) -> None:
        """Initialize cell manager with grid boundaries."""
        # Singles court X boundaries (3 divisions)
        self.court_x_bounds = [
            -HALF_SINGLES_WIDTH,
            -HALF_SINGLES_WIDTH / 3,
            HALF_SINGLES_WIDTH / 3,
            HALF_SINGLES_WIDTH,
        ]

        # Court Y boundaries (3 divisions, for one half)
        half_y = HALF_LENGTH
        self.court_y_bounds = [
            0.0,
            half_y / 3,
            2 * half_y / 3,
            half_y,
        ]

        # Extended boundaries for exterior cells
        self.fence_x_bounds = [
            X_MIN,
            -HALF_SINGLES_WIDTH,
            -HALF_SINGLES_WIDTH / 3,
            HALF_SINGLES_WIDTH / 3,
            HALF_SINGLES_WIDTH,
            X_MAX,
        ]

        # Y bounds extended to fence (5 rows: 3 court + 1 baseline exterior)
        self.fence_y_bounds = [
            0.0,
            HALF_LENGTH / 3,
            2 * HALF_LENGTH / 3,
            HALF_LENGTH,
            abs(Y_MAX),  # Fence Y
        ]

    def position_to_cell_id(self, pos: Tensor, side: str) -> int:
        """Convert position to cell ID (0-19).

        Args:
            pos: Position [3] (x, y, z).
            side: "near" or "far" - which side the ball is targeting.

        Returns:
            int: Cell ID (0-19).

        """
        x = pos[0].item()
        y = pos[1].item()

        # Flip y for near side (make positive for easier calculation)
        if side == "near":
            y = -y

        # Determine x region (0=left-ext, 1-3=court, 4=right-ext)
        x_region = self._get_x_region(x)

        # Determine y region (0-2=court rows, 3=baseline exterior)
        y_region = self._get_y_region(y)

        # Map to cell ID
        return self._region_to_cell_id(x_region, y_region)

    def _get_x_region(self, x: float) -> int:
        """Get x region index (0-4)."""
        bounds = self.fence_x_bounds
        for i in range(len(bounds) - 1):
            if bounds[i] <= x < bounds[i + 1]:
                return i
        return len(bounds) - 2  # Clamp to last region

    def _get_y_region(self, y: float) -> int:
        """Get y region index (0-3), y should be positive."""
        bounds = self.fence_y_bounds
        for i in range(len(bounds) - 1):
            if bounds[i] <= y < bounds[i + 1]:
                return i
        return len(bounds) - 2  # Clamp to last region

    def _region_to_cell_id(self, x_region: int, y_region: int) -> int:
        """Map (x_region, y_region) to cell ID.

        x_region: 0=left-ext, 1-3=court columns, 4=right-ext
        y_region: 0-2=court rows (net to baseline), 3=baseline exterior
        """
        # Court interior (cells 0-8)
        if 1 <= x_region <= 3 and 0 <= y_region <= 2:
            col = x_region - 1  # 0, 1, 2
            row = y_region  # 0, 1, 2
            return row * 3 + col

        # Left exterior (cells 9-11)
        if x_region == 0 and 0 <= y_region <= 2:
            return 9 + y_region

        # Right exterior (cells 12-14)
        if x_region == 4 and 0 <= y_region <= 2:
            return 12 + y_region

        # Baseline exterior (cells 15-19)
        if y_region == 3:
            return 15 + x_region

        # Fallback (should not happen)
        return 0

    def cell_id_to_bounds(self, cell_id: int, side: str) -> CellBounds:
        """Get bounds for a cell ID.

        Args:
            cell_id: Cell ID (0-19).
            side: "near" or "far".

        Returns:
            CellBounds: (x_min, x_max, y_min, y_max) in world coordinates.

        """
        x_min, x_max, y_min, y_max = self._cell_id_to_raw_bounds(cell_id)

        # Flip y for near side
        if side == "near":
            y_min, y_max = -y_max, -y_min

        return CellBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def _cell_id_to_raw_bounds(self, cell_id: int) -> tuple[float, float, float, float]:
        """Get raw bounds (far side orientation)."""
        fence_x = self.fence_x_bounds
        fence_y = self.fence_y_bounds

        # Court interior (0-8)
        if 0 <= cell_id <= 8:
            row = cell_id // 3  # 0, 1, 2
            col = cell_id % 3  # 0, 1, 2
            x_min = fence_x[col + 1]  # Skip left exterior
            x_max = fence_x[col + 2]
            y_min = fence_y[row]
            y_max = fence_y[row + 1]
            return x_min, x_max, y_min, y_max

        # Left exterior (9-11)
        if 9 <= cell_id <= 11:
            row = cell_id - 9
            return fence_x[0], fence_x[1], fence_y[row], fence_y[row + 1]

        # Right exterior (12-14)
        if 12 <= cell_id <= 14:
            row = cell_id - 12
            return fence_x[4], fence_x[5], fence_y[row], fence_y[row + 1]

        # Baseline exterior (15-19)
        if 15 <= cell_id <= 19:
            col = cell_id - 15
            return fence_x[col], fence_x[col + 1], fence_y[3], fence_y[4]

        # Fallback
        return 0.0, 0.0, 0.0, 0.0

    def sample_position_in_cell(
        self,
        cell_id: int,
        side: str,
        z_range: tuple[float, float] = (0.8, 1.4),
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Sample a random position within a cell.

        Args:
            cell_id: Cell ID (0-19).
            side: "near" or "far".
            z_range: (z_min, z_max) height range.
            device: Torch device.

        Returns:
            Tensor: Position [3] (x, y, z).

        """
        bounds = self.cell_id_to_bounds(cell_id, side)

        x = bounds.x_min + torch.rand(1).item() * (bounds.x_max - bounds.x_min)
        y = bounds.y_min + torch.rand(1).item() * (bounds.y_max - bounds.y_min)
        z = z_range[0] + torch.rand(1).item() * (z_range[1] - z_range[0])

        return torch.tensor([x, y, z], device=device)

    def is_in_court(self, cell_id: int) -> bool:
        """Check if cell ID is in court interior (0-8).

        Args:
            cell_id: Cell ID.

        Returns:
            bool: True if cell 0-8.

        """
        return 0 <= cell_id <= 8

    def classify_shot(
        self,
        hit_net_before_bounce: bool,
        hit_fence_before_bounce: bool,
        bounce_pos: Tensor | None,
        target_side: str,
    ) -> tuple[ShotCategory, int | None]:
        """Classify shot into category and to_cell.

        Args:
            hit_net_before_bounce: Whether ball hit net before first bounce.
            hit_fence_before_bounce: Whether ball reached fence before first bounce.
            bounce_pos: Position of first bounce (None if no bounce).
            target_side: "near" or "far" - target side for the shot.

        Returns:
            tuple: (ShotCategory, to_cell or None)

        """
        if hit_net_before_bounce:
            return ShotCategory.DIRECT_NET, None

        if hit_fence_before_bounce:
            return ShotCategory.DIRECT_FENCE, None

        if bounce_pos is None:
            # No bounce occurred (shouldn't happen normally)
            return ShotCategory.DIRECT_FENCE, None

        to_cell = self.position_to_cell_id(bounce_pos, target_side)

        if self.is_in_court(to_cell):
            return ShotCategory.IN_COURT, to_cell
        else:
            return ShotCategory.OUT_COURT, to_cell

    def get_all_cell_ids(self) -> list[int]:
        """Get all valid cell IDs.

        Returns:
            list: Cell IDs 0-19.

        """
        return list(range(20))

    def get_court_cell_ids(self) -> list[int]:
        """Get court interior cell IDs.

        Returns:
            list: Cell IDs 0-8.

        """
        return list(range(9))

    def get_exterior_cell_ids(self) -> list[int]:
        """Get exterior cell IDs.

        Returns:
            list: Cell IDs 9-19.

        """
        return list(range(9, 20))

    def get_cell_center(
        self,
        cell_id: int,
        side: str,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Get the center position of a cell.

        Args:
            cell_id: Cell ID (0-19).
            side: "near" or "far".
            device: Torch device.

        Returns:
            Tensor: Center position [3] (x, y, z=0).

        """
        bounds = self.cell_id_to_bounds(cell_id, side)

        x = (bounds.x_min + bounds.x_max) / 2
        y = (bounds.y_min + bounds.y_max) / 2

        return torch.tensor([x, y, 0.0], device=device)

    def sample_bounce_position_in_cell(
        self,
        cell_id: int,
        side: str,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Sample a random ground-level position within a cell for targeting.

        Unlike sample_position_in_cell which includes height, this returns z=0
        for use as a bounce target position.

        Args:
            cell_id: Cell ID (0-19).
            side: "near" or "far".
            device: Torch device.

        Returns:
            Tensor: Position [3] (x, y, z=0).

        """
        bounds = self.cell_id_to_bounds(cell_id, side)

        x = bounds.x_min + torch.rand(1).item() * (bounds.x_max - bounds.x_min)
        y = bounds.y_min + torch.rand(1).item() * (bounds.y_max - bounds.y_min)

        return torch.tensor([x, y, 0.0], device=device)
