"""Deterministic group-level dataset splitting utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupSplitConfig:
    """Ratios and seed for deterministic group-level splitting."""

    val_ratio: float
    test_ratio: float
    seed: int


def make_group_split_map(
    group_weights: Mapping[str, int],
    config: GroupSplitConfig,
) -> dict[str, str]:
    """Assign complete groups while balancing group and sample counts."""
    if not group_weights:
        return {}
    if (
        config.val_ratio < 0
        or config.test_ratio < 0
        or config.val_ratio + config.test_ratio >= 1
    ):
        raise ValueError("Split ratios must be non-negative and sum to less than 1.")

    keys = list(group_weights)
    group_count = len(keys)

    def requested_count(ratio: float) -> int:
        if ratio == 0 or group_count < 3:
            return 0
        return max(1, int(round(group_count * ratio)))

    split_group_counts = {
        "test": requested_count(config.test_ratio),
        "val": requested_count(config.val_ratio),
    }
    while sum(split_group_counts.values()) >= group_count:
        larger = max(split_group_counts, key=split_group_counts.__getitem__)
        if split_group_counts[larger] <= 1:
            break
        split_group_counts[larger] -= 1

    hash_rank = {
        key: hashlib.sha1(f"{config.seed}:{key}".encode()).hexdigest() for key in keys
    }
    total_weight = sum(max(int(weight), 1) for weight in group_weights.values())
    ratios = {"test": config.test_ratio, "val": config.val_ratio}
    remaining = set(keys)
    assignments: dict[str, str] = {}
    for split in ("test", "val"):
        target_count = split_group_counts[split]
        target_weight = total_weight * ratios[split]
        selected_weight = 0
        for selected_count in range(target_count):
            step_target = target_weight * (selected_count + 1) / target_count
            key = min(
                remaining,
                key=lambda candidate: (
                    abs(
                        selected_weight
                        + max(int(group_weights[candidate]), 1)
                        - step_target
                    ),
                    hash_rank[candidate],
                ),
            )
            assignments[key] = split
            remaining.remove(key)
            selected_weight += max(int(group_weights[key]), 1)

    assignments.update({key: "train" for key in remaining})
    return assignments


__all__ = ["GroupSplitConfig", "make_group_split_map"]
