from __future__ import annotations

from src.tennis.geometry.skeleton import (
    COCO_BONES,
    RACKET_3_NAMES,
    VITPOSE_17_NAMES,
    all_keypoint_names,
    name_to_index,
)


def test_names_and_indices() -> None:
    assert len(VITPOSE_17_NAMES) == 17
    assert len(RACKET_3_NAMES) == 3
    names = all_keypoint_names()
    assert len(names) == 20
    mapping = name_to_index()
    assert len(mapping) == 20
    # uniqueness
    assert len(set(names)) == 20
    assert set(mapping.keys()) == set(names)


def test_coco_bones_indices_in_range() -> None:
    for i, j in COCO_BONES:
        assert 0 <= i < 17
        assert 0 <= j < 17
        assert i != j

