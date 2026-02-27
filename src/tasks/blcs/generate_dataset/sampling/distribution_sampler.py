"""Distribution control for BLCS dataset generation.

Controls the distribution of shots across:
- from_cell (9 cells x 2 sides = 18 origins)
- category (DIRECT_NET, DIRECT_FENCE, IN_COURT, OUT_COURT)
- to_cell (for IN_COURT and OUT_COURT)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.tasks.blcs.simulation.cell_manager import (
    NUM_CELLS_PER_SIDE,
    NUM_IN_COURT_CELLS,
    NUM_OUT_COURT_CELLS,
    ShotCategory,
)

if TYPE_CHECKING:
    pass


@dataclass
class SamplingConfig:
    """Configuration for distribution-controlled sampling."""

    # Category ratios (must sum to 1.0)
    category_ratios: dict[ShotCategory, float] = field(
        default_factory=lambda: {
            ShotCategory.DIRECT_NET: 0.05,
            ShotCategory.DIRECT_FENCE: 0.05,
            ShotCategory.IN_COURT: 0.60,
            ShotCategory.OUT_COURT: 0.30,
        }
    )

    # Cell weights within categories
    # "uniform" or list of weights
    in_court_cell_weights: str | list[float] = "uniform"  # 6 cells (0-5)
    out_court_cell_weights: str | list[float] = "uniform"  # 3 cells (6-8)

    # Target samples per from_cell
    per_from_cell_samples: int = 100


class DistributionSampler:
    """Controls shot distribution for dataset generation.

    Ensures balanced sampling across:
    - from_cell origins
    - shot categories
    - to_cell destinations (within categories)
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        """Initialize distribution sampler.

        Args:
            config: Sampling configuration.
        """
        self.config = config or SamplingConfig()

        # Validate category ratios
        total = sum(self.config.category_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Category ratios must sum to 1.0, got {total}")

        # Initialize counts: (from_cell, side, category, to_cell) -> count
        self.counts: dict[tuple[int, str, ShotCategory, int | None], int] = defaultdict(
            int
        )

        # Compute target counts per category/cell
        self._compute_targets()

    def _compute_targets(self) -> None:
        """Compute target sample counts per category and cell."""
        cfg = self.config
        total = cfg.per_from_cell_samples

        # Targets per category
        self.category_targets = {}
        for cat, ratio in cfg.category_ratios.items():
            self.category_targets[cat] = int(total * ratio)

        # Cell weights for IN_COURT (NUM_IN_COURT_CELLS = 6 cells)
        if cfg.in_court_cell_weights == "uniform":
            self.in_court_weights = [1.0 / NUM_IN_COURT_CELLS] * NUM_IN_COURT_CELLS
        else:
            self.in_court_weights = cfg.in_court_cell_weights

        # Cell weights for OUT_COURT (NUM_OUT_COURT_CELLS = 3 cells)
        if cfg.out_court_cell_weights == "uniform":
            self.out_court_weights = [1.0 / NUM_OUT_COURT_CELLS] * NUM_OUT_COURT_CELLS
        else:
            self.out_court_weights = cfg.out_court_cell_weights

    def get_target_count(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> int:
        """Get target sample count for a specific combination.

        Args:
            from_cell: Origin cell ID (0-8).
            side: "near" or "far".
            category: Shot category.
            to_cell: Destination cell ID (None for DIRECT_NET/DIRECT_FENCE).

        Returns:
            Target count.
        """
        cat_target = self.category_targets[category]

        if category == ShotCategory.DIRECT_NET or category == ShotCategory.DIRECT_FENCE:
            return cat_target
        elif category == ShotCategory.IN_COURT:
            if to_cell is None or to_cell >= NUM_IN_COURT_CELLS:
                return 0
            cell_weight = self.in_court_weights[to_cell]
            return max(1, int(cat_target * cell_weight))
        else:  # OUT_COURT
            if to_cell is None or to_cell < NUM_IN_COURT_CELLS:
                return 0
            cell_idx = to_cell - NUM_IN_COURT_CELLS  # Map 6-8 to 0-2
            if cell_idx >= NUM_OUT_COURT_CELLS:
                return 0
            cell_weight = self.out_court_weights[cell_idx]
            return max(1, int(cat_target * cell_weight))

    def get_current_count(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> int:
        """Get current sample count for a specific combination."""
        key = (from_cell, side, category, to_cell)
        return self.counts[key]

    def should_accept(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> bool:
        """Determine if a shot should be accepted based on distribution."""
        current = self.get_current_count(from_cell, side, category, to_cell)
        target = self.get_target_count(from_cell, side, category, to_cell)
        return current < target

    def record_sample(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> None:
        """Record an accepted sample."""
        key = (from_cell, side, category, to_cell)
        self.counts[key] += 1

    def is_from_cell_complete(self, from_cell: int, side: str) -> bool:
        """Check if all categories for a from_cell are complete."""
        for cat in ShotCategory:
            if cat in (ShotCategory.DIRECT_NET, ShotCategory.DIRECT_FENCE):
                current = self.get_current_count(from_cell, side, cat, None)
                target = self.get_target_count(from_cell, side, cat, None)
                if current < target:
                    return False
            elif cat == ShotCategory.IN_COURT:
                for cell in range(NUM_IN_COURT_CELLS):
                    current = self.get_current_count(from_cell, side, cat, cell)
                    target = self.get_target_count(from_cell, side, cat, cell)
                    if current < target:
                        return False
            elif cat == ShotCategory.OUT_COURT:
                for cell in range(NUM_IN_COURT_CELLS, NUM_CELLS_PER_SIDE):
                    current = self.get_current_count(from_cell, side, cat, cell)
                    target = self.get_target_count(from_cell, side, cat, cell)
                    if current < target:
                        return False
        return True

    def get_statistics(self) -> dict:
        """Get sampling statistics."""
        total_samples = sum(self.counts.values())

        cat_counts = {}
        for cat in ShotCategory:
            count = sum(v for k, v in self.counts.items() if k[2] == cat)
            cat_counts[cat.value] = count

        return {
            "total_samples": total_samples,
            "category_counts": cat_counts,
        }

    def reset(self) -> None:
        """Reset all counts."""
        self.counts.clear()
