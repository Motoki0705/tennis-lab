from __future__ import annotations

import numpy as np

from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.person_sources import (
    source_counts,
    source_variants,
)


def archive(boxes: list[list[float]], scores: list[float]) -> DetectionArchive:
    return DetectionArchive(np.array([0, len(scores)], np.int64), np.asarray(boxes, np.float32).reshape(-1, 4),
                            np.asarray(scores, np.float32), np.array([10.], np.float64))


def test_thresholds_preserve_full_frame_and_union_includes_large_coco_people() -> None:
    ft = archive([[1, 1, 10, 20], [400, 1, 420, 200]], [.01, .8])
    coco = archive([[401, 1, 421, 200], [50, 1, 80, 200]], [.99, .5])
    variants = source_variants({'ft_base': ft}, coco)
    assert set(variants) == {*(f'ft_base_{v:.2f}' for v in (.01, .02, .05, .1, .3)), 'coco_0.30', 'union_0.30'}
    assert len(variants['ft_base_0.01'].scores) == 2
    assert len(variants['ft_base_0.02'].scores) == 1
    union = variants['union_0.30']
    np.testing.assert_array_equal(union.boxes, [[400, 1, 420, 200], [50, 1, 80, 200]])
    np.testing.assert_allclose(union.scores, [.8, .5])  # no cross-model score competition


def test_near_far_units_and_burden_keep_unknowns_and_outside_roi() -> None:
    detection = archive([[1, 1, 10, 20], [1, 200, 10, 230], [200, 1, 230, 40]], [.01, .9, .5])
    camera = CameraLabels(np.array([0, 0], np.int64), np.array([0, 1], np.int64), detection.boxes[:2].astype(np.float64))
    labels = ClipLabels('dev', 1, (LabelledPerson('A', 'player', ''), LabelledPerson('B', 'player', '')), {'cam0': camera}, {})
    counts = source_counts(detection, labels, 'cam0', [[0, 0], [100, 0], [100, 300], [0, 300]])
    assert counts['all']['persons'] == 3 and counts['all']['outside'] == 1
    assert counts['far']['persons'] == 2 and counts['far']['player_hit_03'] == 1
    assert counts['near']['player_hit_05'] == 1
