"""Contract tests for BLCS court-cell mapping."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.cell_manager import (
    CellManager,
    ShotCategory,
)


def test_position_mapping_rejects_the_wrong_canonical_side() -> None:
    manager = CellManager()

    with pytest.raises(ValueError, match="requested canonical court side"):
        manager.position_to_cell_id(
            torch.tensor([0.0, -1.0, 0.0]),
            side="far",
        )


def test_shot_classification_records_own_side_bounce_without_cell_fallback() -> None:
    category, cell = CellManager().classify_shot(
        hit_net_before_bounce=False,
        hit_fence_before_bounce=False,
        bounce_pos=torch.tensor([0.0, -1.0, 0.0]),
        target_side="far",
    )

    assert category is ShotCategory.OUT_COURT
    assert cell is None
