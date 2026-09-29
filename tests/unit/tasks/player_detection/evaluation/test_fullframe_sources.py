from __future__ import annotations

import numpy as np
import pytest

from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.fullframe_sources import (
    SOURCES,
    fullframe_variants,
)


def test_all_sources_keep_fullframe_and_union_uses_both_thresholds() -> None:
    ft = DetectionArchive(np.array([0, 3], np.int64), np.array([[0, 0, 20, 40], [200, 0, 220, 40], [500, 0, 550, 100]], np.float32),
        np.array([.01, .05, .3], np.float32), np.array([1.], np.float64))
    coco = DetectionArchive(np.array([0, 3], np.int64), np.array([[501, 0, 551, 100], [1500, 0, 1600, 200], [1000, 0, 1020, 30]], np.float32),
        np.array([.9, .3, .1], np.float32), np.array([2.], np.float64))
    result = fullframe_variants(ft, coco)
    assert set(result) == set(SOURCES)
    assert [len(result[f'ft_base_{v:.2f}'].scores) for v in (.01, .02, .05)] == [3, 2, 2]
    assert [len(result[f'coco_fullframe_{v:.2f}'].scores) for v in (.05, .1, .3)] == [3, 3, 2]
    np.testing.assert_array_equal(result['union_fullframe_0.30'].boxes, [[500, 0, 550, 100], [1500, 0, 1600, 200]])
    np.testing.assert_allclose(result['union_fullframe_0.30'].scores, [.3, .3])
    longer = DetectionArchive(np.array([0, 0, 3], np.int64), coco.boxes, coco.scores, np.ones(2, np.float64))
    with pytest.raises(ValueError, match='timelines'):
        fullframe_variants(ft, longer)
