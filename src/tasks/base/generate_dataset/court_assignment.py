"""Deterministic balanced target-court assignment shared by domain datasets."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.synthetic_data_generation.scene_contract import MultiCourtLayout


@dataclass(frozen=True, slots=True)
class CourtAssignment:
    """One scene's explicit target court and split."""

    scene_id: str
    split: str
    court_instance_id: str
    candidate_id: str
    selection_seed: int


def assign_courts_balanced(
    scene_splits: Mapping[str, str],
    *,
    layout: MultiCourtLayout,
    seed: int,
) -> tuple[CourtAssignment, ...]:
    """Stable-shuffle scenes and courts, then round-robin continuously by split."""
    if not scene_splits:
        raise ValueError("scene_splits must not be empty.")
    if any(not scene_id.strip() or not split.strip() for scene_id, split in scene_splits.items()):
        raise ValueError("scene IDs and split names must be non-empty.")
    court_ids = sorted(court.court_instance_id for court in layout.courts)
    rng = random.Random(seed)
    rng.shuffle(court_ids)
    assignments: list[CourtAssignment] = []
    offset = 0
    for split in sorted(set(scene_splits.values())):
        scene_ids = sorted(scene_id for scene_id, value in scene_splits.items() if value == split)
        split_rng = random.Random(f"{seed}:{split}")
        split_rng.shuffle(scene_ids)
        for index, scene_id in enumerate(scene_ids):
            court_id = court_ids[(offset + index) % len(court_ids)]
            court = layout.court(court_id)
            assignments.append(
                CourtAssignment(
                    scene_id=scene_id,
                    split=split,
                    court_instance_id=court_id,
                    candidate_id=court.candidate_id,
                    selection_seed=seed,
                )
            )
        offset = (offset + len(scene_ids)) % len(court_ids)
    assignments.sort(key=lambda item: item.scene_id)
    _validate_balance(assignments, layout=layout, scene_splits=scene_splits)
    return tuple(assignments)


def _validate_balance(
    assignments: Sequence[CourtAssignment],
    *,
    layout: MultiCourtLayout,
    scene_splits: Mapping[str, str],
) -> None:
    court_ids = {court.court_instance_id for court in layout.courts}
    if {item.scene_id for item in assignments} != set(scene_splits):
        raise ValueError("Court assignment must cover every scene exactly once.")
    global_counts = Counter(item.court_instance_id for item in assignments)
    if len(assignments) >= len(court_ids) and set(global_counts) != court_ids:
        raise ValueError("Every accepted court must be used when enough scenes exist.")
    counts = [global_counts[court_id] for court_id in court_ids]
    if max(counts) - min(counts) > 1:
        raise ValueError("Global court assignment count difference exceeds one.")
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for item in assignments:
        by_split[item.split][item.court_instance_id] += 1
    for split, split_counts in by_split.items():
        values = [split_counts[court_id] for court_id in court_ids]
        if max(values) - min(values) > 1:
            raise ValueError(f"Court assignment is not balanced within split {split!r}.")


__all__ = ["CourtAssignment", "assign_courts_balanced"]
