"""Cell management for BLCS shot distribution control.

Manages 9-cell grid per side (18 total) for both from_cell and to_cell:

In-court cells (0-5):
  - 0: Deuce Service Box
  - 1: Ad Service Box
  - 2: Deuce Back Court (singles baseline - service line)
  - 3: Ad Back Court (singles baseline - service line)
  - 4: Deuce Doubles Alley
  - 5: Ad Doubles Alley

Out-court cells (6-8):
  - 6: Deuce Side Out (outside doubles, net-to-baseline)
  - 7: Ad Side Out (outside doubles, net-to-baseline)
  - 8: Behind Baseline Out (baseline-to-fence, full width)

Shot category classification:
  - DIRECT_NET: Ball hits net before first bounce
  - DIRECT_FENCE: Ball reaches fence before first bounce
  - IN_COURT: First bounce in doubles court (cells 0-5)
  - OUT_COURT: First bounce outside doubles court (cells 6-8)
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
    SERVICE_LINE_DISTANCE,
    X_MAX,
    X_MIN,
    Y_MAX,
)

if TYPE_CHECKING:
    pass

# ------------------------------------------------------------------
# Cell grid constants
# ------------------------------------------------------------------

NUM_IN_COURT_CELLS: int = 6
NUM_OUT_COURT_CELLS: int = 3
NUM_CELLS_PER_SIDE: int = NUM_IN_COURT_CELLS + NUM_OUT_COURT_CELLS  # 9
NUM_TOTAL_CELLS: int = NUM_CELLS_PER_SIDE * 2  # 18


class ShotCategory(Enum):
    """Shot outcome category."""

    DIRECT_NET = "direct_net"  # Hit net before first bounce
    DIRECT_FENCE = "direct_fence"  # Reached fence before first bounce
    IN_COURT = "in_court"  # First bounce in doubles court
    OUT_COURT = "out_court"  # First bounce outside doubles court


@dataclass
class CellBounds:
    """Axis-aligned bounding rectangle (world coordinates)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class CellManager:
    """Manages 9-cell grid per side for shot distribution control.

    Cell layout (viewed from above, far-side canonical orientation y > 0):

    Fence boundary
    +-------------+-----------------------------+-------------+
    |                        Cell 8                           |
    |                  Behind Baseline Out                    |
    +-----+-------+--------------+--------------+-------+-----+
    |     |       |   Cell 2     |   Cell 3     |       |     |
    |  6  |  4    | Deuce Back   |   Ad Back    |   5   |  7  |
    | D.  | D.    |   Court      |   Court      |  A.   | A.  |
    |Side |Alley  +--------------+--------------+ Alley |Side |
    | Out |       |   Cell 0     |   Cell 1     |       | Out |
    |     |       | Deuce Svc    |   Ad Svc     |       |     |
    |     |       |   Box        |   Box        |       |     |
    +-----+-------+--------------+--------------+-------+-----+
                          Net (y = 0)

    Notes:
    - Cell IDs are shared across near/far sides (0-8 each side).
    - Canonical orientation is the far side.
    - For near-side mapping, both x and y are mirrored before lookup so
      deuce/ad semantics remain side-consistent.
    """

    # Court geometry (precomputed for fast lookup)
    _xs: float = HALF_SINGLES_WIDTH  # 4.115
    _xd: float = HALF_DOUBLES_WIDTH  # 5.485
    _ys: float = SERVICE_LINE_DISTANCE  # 6.40
    _yB: float = HALF_LENGTH  # 11.885
    _x_min: float = X_MIN  # -9.145
    _x_max: float = X_MAX  # +9.145
    _y_max: float = abs(Y_MAX)  # 18.285

    def __init__(self) -> None:
        """Initialize cell manager with geometry boundaries."""
        # Build cell bounds table (far side orientation, y >= 0)
        self._cell_bounds_raw: list[tuple[float, float, float, float]] = [
            # In-court cells
            (-self._xs, 0.0, 0.0, self._ys),  # 0: Deuce Svc Box
            (0.0, self._xs, 0.0, self._ys),  # 1: Ad Svc Box
            (-self._xs, 0.0, self._ys, self._yB),  # 2: Deuce Back Court
            (0.0, self._xs, self._ys, self._yB),  # 3: Ad Back Court
            (-self._xd, -self._xs, 0.0, self._yB),  # 4: Deuce Doubles Alley
            (self._xs, self._xd, 0.0, self._yB),  # 5: Ad Doubles Alley
            # Out-court cells
            (self._x_min, -self._xd, 0.0, self._yB),  # 6: Deuce Side Out
            (self._xd, self._x_max, 0.0, self._yB),  # 7: Ad Side Out
            (self._x_min, self._x_max, self._yB, self._y_max),  # 8: Behind Baseline
        ]

    # ------------------------------------------------------------------
    # Position -> Cell mapping
    # ------------------------------------------------------------------

    def position_to_cell_id(self, pos: Tensor, side: str) -> int:
        """Convert world position to cell ID (0-8).

        Args:
            pos: Position [3] (x, y, z).
            side: ``"near"`` or ``"far"`` -- target side.

        Returns:
            Cell ID (0-8).
        """
        if side not in ("near", "far"):
            raise ValueError(f"side must be 'near' or 'far', got {side!r}")

        x = pos[0].item()
        y = pos[1].item()

        # Normalise to far-side canonical orientation.
        # Mirror both x and y for near side to preserve deuce/ad semantics.
        if side == "near":
            x = -x
            y = -y

        return self._xy_to_cell(x, y)

    def _xy_to_cell(self, x: float, y: float) -> int:
        """Map (x, y) in far-side orientation to cell ID."""
        # Check in-court regions first (most common)
        if 0.0 <= y < self._ys:
            # Service box row
            if -self._xs <= x < 0.0:
                return 0  # Deuce Svc Box
            if 0.0 <= x <= self._xs:
                return 1  # Ad Svc Box
            if -self._xd <= x < -self._xs:
                return 4  # Deuce Doubles Alley
            if self._xs < x <= self._xd:
                return 5  # Ad Doubles Alley
        elif self._ys <= y <= self._yB:
            # Back court row
            if -self._xs <= x < 0.0:
                return 2  # Deuce Back Court
            if 0.0 <= x <= self._xs:
                return 3  # Ad Back Court
            if -self._xd <= x < -self._xs:
                return 4  # Deuce Doubles Alley
            if self._xs < x <= self._xd:
                return 5  # Ad Doubles Alley

        # Out-court
        if y > self._yB:
            return 8  # Behind Baseline
        if x < -self._xd:
            return 6  # Deuce Side Out
        if x > self._xd:
            return 7  # Ad Side Out

        raise ValueError(
            "Position does not lie on the requested canonical court side: "
            f"x={x}, y={y}."
        )

    # ------------------------------------------------------------------
    # Cell -> Bounds mapping
    # ------------------------------------------------------------------

    def cell_id_to_bounds(self, cell_id: int, side: str) -> CellBounds:
        """Get world-coordinate bounds for a cell ID.

        Args:
            cell_id: Cell ID (0-8).
            side: ``"near"`` or ``"far"``.

        Returns:
            CellBounds in world coordinates.
        """
        if not 0 <= cell_id < NUM_CELLS_PER_SIDE:
            raise ValueError(
                f"cell_id must be in [0, {NUM_CELLS_PER_SIDE - 1}], got {cell_id}"
            )
        if side not in ("near", "far"):
            raise ValueError(f"side must be 'near' or 'far', got {side!r}")

        x_min, x_max, y_min, y_max = self._cell_bounds_raw[cell_id]

        if side == "near":
            x_min, x_max = -x_max, -x_min
            y_min, y_max = -y_max, -y_min

        return CellBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample_position_in_cell(
        self,
        cell_id: int,
        side: str,
        z_range: tuple[float, float] = (0.8, 1.4),
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Sample a random position within a cell (with height).

        Args:
            cell_id: Cell ID (0-8).
            side: ``"near"`` or ``"far"``.
            z_range: (z_min, z_max) height range in metres.
            device: Torch device.

        Returns:
            Position [3] (x, y, z).
        """
        bounds = self.cell_id_to_bounds(cell_id, side)

        x = bounds.x_min + torch.rand(1).item() * (bounds.x_max - bounds.x_min)
        y = bounds.y_min + torch.rand(1).item() * (bounds.y_max - bounds.y_min)
        z = z_range[0] + torch.rand(1).item() * (z_range[1] - z_range[0])

        return torch.tensor([x, y, z], device=device)

    def sample_bounce_position_in_cell(
        self,
        cell_id: int,
        side: str,
        device: str | torch.device = "cpu",
        margin: float = 0.0,
    ) -> Tensor:
        """Sample a random ground-level (z=0) position within a cell.

        Args:
            cell_id: Cell ID (0-8).
            side: ``"near"`` or ``"far"``.
            device: Torch device.
            margin: Inset (metres) from each cell edge. Falls back to the cell
                centre on an axis whose extent is smaller than ``2 * margin``.

        Returns:
            Position [3] (x, y, z=0).
        """
        bounds = self.cell_id_to_bounds(cell_id, side)

        x = self._sample_in_range(bounds.x_min, bounds.x_max, margin)
        y = self._sample_in_range(bounds.y_min, bounds.y_max, margin)

        return torch.tensor([x, y, 0.0], device=device)

    @staticmethod
    def _sample_in_range(lo: float, hi: float, margin: float) -> float:
        """Sample uniformly in [lo + margin, hi - margin] with centre fallback."""
        lo_m = lo + margin
        hi_m = hi - margin
        if lo_m >= hi_m:
            return (lo + hi) / 2.0
        return float(lo_m + torch.rand(1).item() * (hi_m - lo_m))

    def get_cell_center(
        self,
        cell_id: int,
        side: str,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Get the centre position of a cell (z=0).

        Args:
            cell_id: Cell ID (0-8).
            side: ``"near"`` or ``"far"``.
            device: Torch device.

        Returns:
            Centre [3] (x, y, z=0).
        """
        bounds = self.cell_id_to_bounds(cell_id, side)
        x = (bounds.x_min + bounds.x_max) / 2
        y = (bounds.y_min + bounds.y_max) / 2
        return torch.tensor([x, y, 0.0], device=device)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def is_in_court(self, cell_id: int) -> bool:
        """Check if cell ID is in-court (0-5).

        Args:
            cell_id: Cell ID.

        Returns:
            True if in-court.
        """
        return 0 <= cell_id < NUM_IN_COURT_CELLS

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
            target_side: ``"near"`` or ``"far"``.

        Returns:
            Tuple of (ShotCategory, to_cell or None).
        """
        if hit_net_before_bounce:
            return ShotCategory.DIRECT_NET, None

        if hit_fence_before_bounce:
            return ShotCategory.DIRECT_FENCE, None

        if bounce_pos is None:
            return ShotCategory.DIRECT_FENCE, None

        y = float(bounce_pos[1].item())
        is_target_side = y > 0.0 if target_side == "far" else y < 0.0
        if not is_target_side:
            return ShotCategory.OUT_COURT, None

        to_cell = self.position_to_cell_id(bounce_pos, target_side)

        if self.is_in_court(to_cell):
            return ShotCategory.IN_COURT, to_cell
        else:
            return ShotCategory.OUT_COURT, to_cell

    # ------------------------------------------------------------------
    # Enumerators
    # ------------------------------------------------------------------

    def get_all_cell_ids(self) -> list[int]:
        """Get all valid cell IDs (0-8)."""
        return list(range(NUM_CELLS_PER_SIDE))

    def get_court_cell_ids(self) -> list[int]:
        """Get in-court cell IDs (0-5)."""
        return list(range(NUM_IN_COURT_CELLS))

    def get_exterior_cell_ids(self) -> list[int]:
        """Get out-court cell IDs (6-8)."""
        return list(range(NUM_IN_COURT_CELLS, NUM_CELLS_PER_SIDE))

    def get_service_box_cell_ids(self) -> list[int]:
        """Get service box cell IDs (0=deuce, 1=ad)."""
        return [0, 1]
