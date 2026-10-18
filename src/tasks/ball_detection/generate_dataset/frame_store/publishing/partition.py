"""Resolve a frozen, label-independent source split for a complete publication."""

from __future__ import annotations

from collections.abc import Mapping, Set

from src.tasks.base.annotation.contracts import content_digest

from .contracts import BallPublicationPolicy, Split


def publication_splits(
    policy: BallPublicationPolicy,
    existing: Mapping[str, Split],
    added_sources: Set[str],
    *,
    expected_new_sources: int,
) -> dict[str, Split]:
    splits = dict(existing)
    if policy.source_partition is None:
        for entry in policy.source_splits:
            if entry.source_id in splits and splits[entry.source_id] != entry.split:
                raise ValueError(
                    "publication policy cannot change an existing source split"
                )
            splits[entry.source_id] = entry.split
    else:
        recipe = policy.source_partition
        sources = added_sources - existing.keys()
        if len(sources) != expected_new_sources:
            raise ValueError(
                "Source partition requires the complete planned set of new videos"
            )
        weighted: tuple[tuple[Split, int], ...] = (
            ("train", recipe.train_weight),
            ("val", recipe.validation_weight),
            ("test", recipe.test_weight),
        )
        total = sum(weight for _, weight in weighted)
        if any(len(sources) * weight % total for _, weight in weighted):
            raise ValueError(
                "Video count cannot represent the exact source split weights"
            )
        ranked = sorted(
            sources,
            key=lambda source: (
                content_digest(["ball.source-partition.v1", recipe.seed, source]),
                source,
            ),
        )
        start = 0
        for split, weight in weighted:
            stop = start + len(sources) * weight // total
            splits.update((source, split) for source in ranked[start:stop])
            start = stop
    if added_sources - splits.keys():
        raise ValueError("every new source requires an explicit fixed split")
    return splits
