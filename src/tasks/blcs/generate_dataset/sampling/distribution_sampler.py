"""Distribution control for BLCS dataset generation.

Controls the distribution of shots across:
- from_cell (20 cells × 2 sides = 40 origins)
- category (DIRECT_NET, DIRECT_FENCE, IN_COURT, OUT_COURT)
- to_cell (for IN_COURT and OUT_COURT)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.blcs.simulation.cell_manager import ShotCategory

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
    in_court_cell_weights: str | list[float] = "uniform"  # 9 cells (0-8)
    out_court_cell_weights: str | list[float] = "uniform"  # 11 cells (9-19)

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
        # to_cell is None for DIRECT_NET and DIRECT_FENCE
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

        # Cell weights for IN_COURT (9 cells)
        if cfg.in_court_cell_weights == "uniform":
            self.in_court_weights = [1.0 / 9] * 9
        else:
            self.in_court_weights = cfg.in_court_cell_weights

        # Cell weights for OUT_COURT (11 cells)
        if cfg.out_court_cell_weights == "uniform":
            self.out_court_weights = [1.0 / 11] * 11
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
            from_cell: Origin cell ID (0-19).
            side: "near" or "far".
            category: Shot category.
            to_cell: Destination cell ID (None for DIRECT_NET/DIRECT_FENCE).

        Returns:
            int: Target count.

        """
        cat_target = self.category_targets[category]

        if category == ShotCategory.DIRECT_NET or category == ShotCategory.DIRECT_FENCE:
            return cat_target
        elif category == ShotCategory.IN_COURT:
            if to_cell is None or to_cell > 8:
                return 0
            cell_weight = self.in_court_weights[to_cell]
            return max(1, int(cat_target * cell_weight))
        else:  # OUT_COURT
            if to_cell is None or to_cell < 9:
                return 0
            cell_idx = to_cell - 9  # Map 9-19 to 0-10
            cell_weight = self.out_court_weights[cell_idx]
            return max(1, int(cat_target * cell_weight))

    def get_current_count(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> int:
        """Get current sample count for a specific combination.

        Args:
            from_cell: Origin cell ID.
            side: "near" or "far".
            category: Shot category.
            to_cell: Destination cell ID.

        Returns:
            int: Current count.

        """
        key = (from_cell, side, category, to_cell)
        return self.counts[key]

    def should_accept(
        self,
        from_cell: int,
        side: str,
        category: ShotCategory,
        to_cell: int | None,
    ) -> bool:
        """Determine if a shot should be accepted based on distribution.

        Args:
            from_cell: Origin cell ID.
            side: "near" or "far".
            category: Shot category.
            to_cell: Destination cell ID.

        Returns:
            bool: True if shot should be accepted.

        """
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
        """Record an accepted sample.

        Args:
            from_cell: Origin cell ID.
            side: "near" or "far".
            category: Shot category.
            to_cell: Destination cell ID.

        """
        key = (from_cell, side, category, to_cell)
        self.counts[key] += 1

    def is_from_cell_complete(self, from_cell: int, side: str) -> bool:
        """Check if all categories for a from_cell are complete.

        Args:
            from_cell: Origin cell ID.
            side: "near" or "far".

        Returns:
            bool: True if all targets reached.

        """
        # Check DIRECT_NET
        if self.should_accept(from_cell, side, ShotCategory.DIRECT_NET, None):
            return False

        # Check DIRECT_FENCE
        if self.should_accept(from_cell, side, ShotCategory.DIRECT_FENCE, None):
            return False

        # Check IN_COURT cells
        for to_cell in range(9):
            if self.should_accept(from_cell, side, ShotCategory.IN_COURT, to_cell):
                return False

        # Check OUT_COURT cells
        for to_cell in range(9, 20):
            if self.should_accept(from_cell, side, ShotCategory.OUT_COURT, to_cell):
                return False

        return True

    def get_completion_ratio(self, from_cell: int, side: str) -> float:
        """Get completion ratio for a from_cell.

        Args:
            from_cell: Origin cell ID.
            side: "near" or "far".

        Returns:
            float: Ratio of completed samples to target.

        """
        total_current = 0
        total_target = 0

        # DIRECT_NET
        total_current += self.get_current_count(
            from_cell, side, ShotCategory.DIRECT_NET, None
        )
        total_target += self.get_target_count(
            from_cell, side, ShotCategory.DIRECT_NET, None
        )

        # DIRECT_FENCE
        total_current += self.get_current_count(
            from_cell, side, ShotCategory.DIRECT_FENCE, None
        )
        total_target += self.get_target_count(
            from_cell, side, ShotCategory.DIRECT_FENCE, None
        )

        # IN_COURT cells
        for to_cell in range(9):
            total_current += self.get_current_count(
                from_cell, side, ShotCategory.IN_COURT, to_cell
            )
            total_target += self.get_target_count(
                from_cell, side, ShotCategory.IN_COURT, to_cell
            )

        # OUT_COURT cells
        for to_cell in range(9, 20):
            total_current += self.get_current_count(
                from_cell, side, ShotCategory.OUT_COURT, to_cell
            )
            total_target += self.get_target_count(
                from_cell, side, ShotCategory.OUT_COURT, to_cell
            )

        if total_target == 0:
            return 1.0

        return total_current / total_target

    def get_statistics(self) -> dict:
        """Get overall sampling statistics.

        Returns:
            dict: Statistics including counts per category, completion ratios, etc.

        """
        stats = {
            "total_samples": sum(self.counts.values()),
            "category_counts": {},
            "from_cell_completion": {},
        }

        # Count per category
        for cat in ShotCategory:
            cat_count = sum(
                count for (_, _, c, _), count in self.counts.items() if c == cat
            )
            stats["category_counts"][cat.value] = cat_count

        # Completion per from_cell
        for side in ["near", "far"]:
            for from_cell in range(20):
                key = f"{side}_{from_cell}"
                stats["from_cell_completion"][key] = self.get_completion_ratio(
                    from_cell, side
                )

        return stats

    def reset(self) -> None:
        """Reset all counts."""
        self.counts.clear()
