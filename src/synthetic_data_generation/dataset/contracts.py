"""Shared semantic dataset manifests without artifact identity fields."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from src.synthetic_data_generation.scene_contract import RigidTransform

_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DatasetDomain(StrEnum):
    """Canonical synthetic dataset domains."""

    COURT = "court"
    BLCS = "blcs"
    PLCS = "plcs"


@dataclass(frozen=True, slots=True)
class TargetCourtBinding:
    """Traceable target-court authority shared by trajectory and cameras."""

    court_instance_id: str
    candidate_id: str
    scene_from_court: RigidTransform
    selection_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("court_instance_id", self.court_instance_id),
            ("candidate_id", self.candidate_id),
        ):
            if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value) is None:
                raise ValueError(f"{name} must be a portable identifier.")
        if not isinstance(self.scene_from_court, RigidTransform):
            raise TypeError("scene_from_court must be a RigidTransform.")
        if isinstance(self.selection_seed, bool) or not isinstance(self.selection_seed, int):
            raise TypeError("selection_seed must be an integer.")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical metadata record."""
        return {
            "court_instance_id": self.court_instance_id,
            "candidate_id": self.candidate_id,
            "scene_from_court": self.scene_from_court.to_list(),
            "selection_seed": self.selection_seed,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse one exact target-court binding."""
        raw = _mapping(value, name="target court binding")
        require_exact_keys(
            raw,
            keys=(
                "court_instance_id",
                "candidate_id",
                "scene_from_court",
                "selection_seed",
            ),
            name="target court binding",
        )
        transform = raw["scene_from_court"]
        if not isinstance(transform, Sequence) or isinstance(transform, (str, bytes)):
            raise TypeError("scene_from_court must contain sixteen numeric values.")
        return cls(
            court_instance_id=_text(raw["court_instance_id"], name="court_instance_id"),
            candidate_id=_text(raw["candidate_id"], name="candidate_id"),
            scene_from_court=RigidTransform(tuple(_number(item, name="scene_from_court") for item in transform)),
            selection_seed=_integer(raw["selection_seed"], name="selection_seed"),
        )


@dataclass(frozen=True, slots=True)
class FrameInventory:
    """Exact source/planned/rendered/labelled frame-set equality contract."""

    source_count: int
    planned_indices: tuple[int, ...]
    rendered_indices: tuple[int, ...]
    labelled_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.source_count, bool) or self.source_count <= 0:
            raise ValueError("source_count must be a positive integer.")
        expected = tuple(range(self.source_count))
        for name, values in (
            ("planned_indices", self.planned_indices),
            ("rendered_indices", self.rendered_indices),
            ("labelled_indices", self.labelled_indices),
        ):
            if values != expected:
                missing = sorted(set(expected) - set(values))
                duplicates = sorted({value for value in values if values.count(value) > 1})
                unexpected = sorted(set(values) - set(expected))
                raise ValueError(
                    f"{name} must equal 0..T-1 in order; missing={missing}, "
                    f"duplicates={duplicates}, unexpected={unexpected}."
                )

    def to_dict(self) -> dict[str, object]:
        """Return compact equality evidence."""
        return {
            "source": self.source_count,
            "planned": len(self.planned_indices),
            "rendered": len(self.rendered_indices),
            "labelled": len(self.labelled_indices),
            "first_frame": 0,
            "last_frame": self.source_count - 1,
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Revalidate compact persisted equality evidence."""
        raw = _mapping(value, name="frame inventory")
        require_exact_keys(
            raw,
            keys=("source", "planned", "rendered", "labelled", "first_frame", "last_frame"),
            name="frame inventory",
        )
        source_count = _integer(raw["source"], name="frame inventory source", minimum=1)
        expected_summary = {
            "planned": source_count,
            "rendered": source_count,
            "labelled": source_count,
            "first_frame": 0,
            "last_frame": source_count - 1,
        }
        observed = {
            name: _integer(raw[name], name=f"frame inventory {name}")
            for name in expected_summary
        }
        if observed != expected_summary:
            raise ValueError(
                "Persisted frame inventory does not prove exact 0..T-1 equality; "
                f"expected={expected_summary}, observed={observed}."
            )
        indices = tuple(range(source_count))
        return cls(source_count, indices, indices, indices)


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Common published dataset metadata consumed by the final report."""

    scene_id: str
    domain: DatasetDomain
    schema: str
    frame_inventory: FrameInventory
    target_courts: tuple[TargetCourtBinding, ...]
    metadata: Mapping[str, object]
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.scene_id.strip() or self.scene_id != self.scene_id.strip():
            raise ValueError("scene_id must be a non-empty trimmed string.")
        if not self.schema.strip() or self.schema != self.schema.strip():
            raise ValueError("schema must be a non-empty trimmed string.")
        if not self.target_courts:
            raise ValueError("A dataset must record at least one target-court binding.")
        if not self.diagnostics or any(not value.strip() for value in self.diagnostics):
            raise ValueError("A dataset must record non-empty diagnostic paths.")
        if len(self.target_courts) != len(
            {binding.court_instance_id for binding in self.target_courts}
        ):
            raise ValueError("Target-court bindings must have unique court_instance_id values.")
        if any(not isinstance(key, str) for key in self.metadata):
            raise TypeError("Dataset metadata keys must be strings.")

    def to_dict(self) -> dict[str, object]:
        """Return the strict shared dataset manifest representation."""
        return {
            "scene_id": self.scene_id,
            "domain": self.domain.value,
            "schema": self.schema,
            "frame_inventory": self.frame_inventory.to_dict(),
            "target_courts": [binding.to_dict() for binding in self.target_courts],
            "metadata": dict(self.metadata),
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: object) -> Self:
        """Parse a strict shared manifest and recompute frame equality."""
        raw = _mapping(value, name="dataset manifest")
        require_exact_keys(
            raw,
            keys=(
                "scene_id",
                "domain",
                "schema",
                "frame_inventory",
                "target_courts",
                "metadata",
                "diagnostics",
            ),
            name="dataset manifest",
        )
        bindings = _sequence(raw["target_courts"], name="target_courts")
        diagnostics = _sequence(raw["diagnostics"], name="diagnostics")
        metadata = _mapping(raw["metadata"], name="metadata")
        try:
            domain = DatasetDomain(_text(raw["domain"], name="domain"))
        except ValueError as error:
            raise ValueError(f"Unknown dataset domain: {raw['domain']!r}.") from error
        return cls(
            scene_id=_text(raw["scene_id"], name="scene_id"),
            domain=domain,
            schema=_text(raw["schema"], name="schema"),
            frame_inventory=FrameInventory.from_dict(raw["frame_inventory"]),
            target_courts=tuple(TargetCourtBinding.from_dict(item) for item in bindings),
            metadata=dict(metadata),
            diagnostics=tuple(_text(item, name="diagnostic path") for item in diagnostics),
        )


def require_exact_keys(
    value: Mapping[str, object],
    *,
    keys: Sequence[str],
    name: str,
) -> None:
    """Reject missing and unknown schema keys at a public mapping boundary."""
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys do not match; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}."
        )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a non-string sequence.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} values must be numeric.")
    return float(value)


__all__ = [
    "DatasetDomain",
    "DatasetManifest",
    "FrameInventory",
    "TargetCourtBinding",
    "require_exact_keys",
]
