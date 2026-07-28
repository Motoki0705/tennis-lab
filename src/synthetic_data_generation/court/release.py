"""Family-disjoint release splits for multi-court orbit datasets."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.synthetic_data_generation.court.orbits import OrbitFamilySpec

DatasetSplit = Literal["train", "validation", "test"]
_FAMILY_PATTERN = re.compile(
    r"^(circle|ellipse)-scale-([0-9]+\.[0-9]{2})-target-(complex|court_[0-9]+)$"
)


@dataclass(frozen=True)
class OrbitFamilyIdentity:
    """Semantic identity used to stratify an entire smooth trajectory."""

    family_id: str
    shape: str
    scale_label: str
    target_label: str

    @classmethod
    def from_spec(cls, family: OrbitFamilySpec) -> OrbitFamilyIdentity:
        """Parse and cross-check one strict orbit-family identifier."""
        match = _FAMILY_PATTERN.fullmatch(family.family_id)
        if match is None:
            raise ValueError(f"Invalid orbit family id: {family.family_id!r}.")
        shape, scale_label, target_label = match.groups()
        expected_target = family.target_court_instance_id or "complex"
        if shape != family.shape or target_label != expected_target:
            raise ValueError("Orbit family id differs from its typed attributes.")
        return cls(
            family_id=family.family_id,
            shape=shape,
            scale_label=scale_label,
            target_label=target_label,
        )


@dataclass(frozen=True)
class FamilySplitRecord:
    """One whole trajectory assigned to exactly one dataset split."""

    family: OrbitFamilyIdentity
    split: DatasetSplit


@dataclass(frozen=True)
class FamilySplitAssignment:
    """Immutable family-level train/validation/test partition."""

    seed: int
    records: tuple[FamilySplitRecord, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Split seed must be a non-negative integer.")
        records = tuple(self.records)
        family_ids = [record.family.family_id for record in records]
        if not records or len(family_ids) != len(set(family_ids)):
            raise ValueError("Every orbit family must occur exactly once.")
        splits = {record.split for record in records}
        if splits != {"train", "validation", "test"}:
            raise ValueError("All three dataset splits must be non-empty.")
        object.__setattr__(self, "records", records)

    def split_for_family(self, family_id: str) -> DatasetSplit:
        """Return the unique split for ``family_id`` or reject unknown input."""
        matches = [
            record.split
            for record in self.records
            if record.family.family_id == family_id
        ]
        if len(matches) != 1:
            raise KeyError(f"Unknown orbit family: {family_id!r}.")
        return matches[0]

    def families_for_split(self, split: DatasetSplit) -> tuple[str, ...]:
        """Return sorted whole-family membership for one split."""
        return tuple(
            sorted(
                record.family.family_id
                for record in self.records
                if record.split == split
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible split contract."""
        return {
            "seed": self.seed,
            "strategy": "greedy-semantic-cover-family-disjoint-v1",
            "records": [
                {
                    "family_id": record.family.family_id,
                    "shape": record.family.shape,
                    "scale_label": record.family.scale_label,
                    "target_label": record.family.target_label,
                    "split": record.split,
                }
                for record in self.records
            ],
        }


def assign_family_disjoint_splits(
    families: Sequence[OrbitFamilySpec],
    *,
    seed: int,
) -> FamilySplitAssignment:
    """Assign whole orbit families while covering semantics in both holdouts."""
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("Split seed must be a non-negative integer.")
    identities = tuple(OrbitFamilyIdentity.from_spec(family) for family in families)
    if len(identities) != len({identity.family_id for identity in identities}):
        raise ValueError("Orbit family IDs must be unique.")
    universes = (
        {identity.shape for identity in identities},
        {identity.scale_label for identity in identities},
        {identity.target_label for identity in identities},
    )
    holdout_count = max(len(universe) for universe in universes)
    if len(identities) < 2 * holdout_count + 1:
        raise ValueError("Too few families for semantic validation/test holdouts.")

    available = list(identities)
    assigned: dict[str, DatasetSplit] = {}
    for split in ("validation", "test"):
        uncovered: tuple[set[str], set[str], set[str]] = (
            set(universes[0]),
            set(universes[1]),
            set(universes[2]),
        )
        for selection_index in range(holdout_count):
            ranked = sorted(
                available,
                key=lambda identity: (
                    -_semantic_gain(identity, uncovered),
                    _stable_rank(
                        identity.family_id,
                        seed=seed,
                        split=split,
                        selection_index=selection_index,
                    ),
                ),
            )
            selected = ranked[0]
            assigned[selected.family_id] = split
            available.remove(selected)
            for values, attribute in zip(
                uncovered,
                (
                    selected.shape,
                    selected.scale_label,
                    selected.target_label,
                ),
                strict=True,
            ):
                values.discard(attribute)
        if any(uncovered):
            raise ValueError(
                f"{split} cannot cover every shape/scale/target without leakage."
            )
    for identity in available:
        assigned[identity.family_id] = "train"
    records = tuple(
        FamilySplitRecord(
            family=identity,
            split=assigned[identity.family_id],
        )
        for identity in identities
    )
    return FamilySplitAssignment(seed=seed, records=records)


def _semantic_gain(
    identity: OrbitFamilyIdentity,
    uncovered: tuple[set[str], set[str], set[str]],
) -> int:
    return sum(
        attribute in values
        for attribute, values in zip(
            (identity.shape, identity.scale_label, identity.target_label),
            uncovered,
            strict=True,
        )
    )


def _stable_rank(
    family_id: str,
    *,
    seed: int,
    split: str,
    selection_index: int,
) -> str:
    return hashlib.sha256(
        f"{seed}:{split}:{selection_index}:{family_id}".encode()
    ).hexdigest()
