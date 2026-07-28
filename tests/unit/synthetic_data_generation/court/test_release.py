"""Tests for family-disjoint multi-court dataset splits."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.court.orbits import OrbitFamilySpec
from src.synthetic_data_generation.court.release import (
    DatasetSplit,
    assign_family_disjoint_splits,
)


def _families() -> tuple[OrbitFamilySpec, ...]:
    result = []
    for shape in ("circle", "ellipse"):
        for scale in (0.75, 1.0, 1.3):
            for target in (None, "court_0", "court_1"):
                target_name = target or "complex"
                result.append(
                    OrbitFamilySpec(
                        family_id=(
                            f"{shape}-scale-{scale:.2f}-target-{target_name}"
                        ),
                        shape=shape,
                        radius_x_m=10.0 * scale,
                        radius_y_m=10.0 * scale,
                        height_m=2.0,
                        target_court_instance_id=target,
                        phase_radians=0.0,
                        sample_count=24,
                    )
                )
    return tuple(result)


def test_split_is_family_disjoint_deterministic_and_stratified() -> None:
    families = _families()
    assignment = assign_family_disjoint_splits(families, seed=26072814)
    repeated = assign_family_disjoint_splits(families, seed=26072814)

    assert assignment == repeated
    assert len(assignment.families_for_split("train")) == 12
    assert len(assignment.families_for_split("validation")) == 3
    assert len(assignment.families_for_split("test")) == 3
    splits: tuple[DatasetSplit, ...] = ("train", "validation", "test")
    sets = [
        set(assignment.families_for_split(split))
        for split in splits
    ]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    for split in splits[1:]:
        records = [
            record
            for record in assignment.records
            if record.split == split
        ]
        assert {record.family.shape for record in records} == {
            "circle",
            "ellipse",
        }
        assert {record.family.scale_label for record in records} == {
            "0.75",
            "1.00",
            "1.30",
        }
        assert {record.family.target_label for record in records} == {
            "complex",
            "court_0",
            "court_1",
        }


def test_all_frames_in_one_family_share_one_split() -> None:
    assignment = assign_family_disjoint_splits(_families(), seed=26072814)
    family_id = "circle-scale-1.00-target-court_0"

    assert len({assignment.split_for_family(family_id) for _ in range(24)}) == 1


def test_split_rejects_mismatched_family_identity() -> None:
    family = OrbitFamilySpec(
        family_id="circle-scale-1.00-target-court_0",
        shape="ellipse",
        radius_x_m=10.0,
        radius_y_m=11.0,
        height_m=2.0,
        target_court_instance_id="court_0",
        phase_radians=0.0,
        sample_count=24,
    )

    with pytest.raises(ValueError, match="typed attributes"):
        assign_family_disjoint_splits((family,) * 7, seed=1)
