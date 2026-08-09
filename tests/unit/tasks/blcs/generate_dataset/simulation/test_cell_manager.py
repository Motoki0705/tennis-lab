"""Contract tests for BLCS court-cell mapping."""

from __future__ import annotations

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager


def test_position_mapping_rejects_the_wrong_canonical_side() -> None:
    manager = CellManager()

    with pytest.raises(ValueError, match="requested canonical court side"):
        manager.position_to_cell_id(
            torch.tensor([0.0, -1.0, 0.0]),
            side="far",
        )
