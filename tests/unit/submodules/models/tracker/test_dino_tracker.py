"""Tests for DINO detection association through Ultralytics BoT-SORT."""

import numpy as np
from numpy.typing import NDArray

from src.submodules.models.dino import PersonDetectionResult
from src.submodules.models.tracker.dino_tracker import BotSortAssociator


def _detections(x_offset: float) -> PersonDetectionResult:
    return PersonDetectionResult(
        boxes_xyxy=np.array(
            [[10 + x_offset, 20, 50 + x_offset, 100]], dtype=np.float32
        ),
        scores=np.array([0.95], dtype=np.float32),
    )


def test_bot_sort_keeps_id_for_adjacent_dino_boxes() -> None:
    associator = BotSortAssociator()
    frame: NDArray[np.uint8] = np.zeros((120, 160, 3), dtype=np.uint8)

    first = associator.update(_detections(0), frame)
    second = associator.update(_detections(2), frame)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["id"] == second[0]["id"]
    np.testing.assert_allclose(second[0]["bbx_xyxy"], [12, 20, 52, 100], atol=2)
