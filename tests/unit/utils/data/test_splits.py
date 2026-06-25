"""Unit tests for :mod:`src.utils.data.splits`."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from src.utils.data.splits import GroupSplitConfig, make_group_split_map


def _uniform_groups(n: int) -> dict[str, int]:
    return {f"g{i}": 1 for i in range(n)}


class TestMakeGroupSplitMap:
    def test_empty_groups_returns_empty(self) -> None:
        config = GroupSplitConfig(val_ratio=0.2, test_ratio=0.2, seed=0)
        assert make_group_split_map({}, config) == {}

    @pytest.mark.parametrize(
        "val,test",
        [(-0.1, 0.2), (0.2, -0.1), (0.5, 0.5), (0.7, 0.4)],
    )
    def test_invalid_ratios_raise(self, val: float, test: float) -> None:
        config = GroupSplitConfig(val_ratio=val, test_ratio=test, seed=0)
        with pytest.raises(ValueError, match="ratios"):
            make_group_split_map(_uniform_groups(10), config)

    def test_fewer_than_three_groups_all_train(self) -> None:
        config = GroupSplitConfig(val_ratio=0.3, test_ratio=0.3, seed=0)
        result = make_group_split_map(_uniform_groups(2), config)
        assert set(result.values()) == {"train"}

    def test_every_group_is_assigned_once(self) -> None:
        config = GroupSplitConfig(val_ratio=0.2, test_ratio=0.2, seed=7)
        groups = _uniform_groups(10)
        result = make_group_split_map(groups, config)
        assert set(result) == set(groups)
        assert set(result.values()) <= {"train", "val", "test"}

    def test_split_group_counts(self) -> None:
        config = GroupSplitConfig(val_ratio=0.2, test_ratio=0.2, seed=7)
        result = make_group_split_map(_uniform_groups(10), config)
        counts = Counter(result.values())
        assert counts["test"] == 2
        assert counts["val"] == 2
        assert counts["train"] == 6

    def test_deterministic_for_same_seed(self) -> None:
        config = GroupSplitConfig(val_ratio=0.2, test_ratio=0.2, seed=42)
        groups = _uniform_groups(12)
        first = make_group_split_map(groups, config)
        second = make_group_split_map(groups, config)
        assert first == second

    def test_seed_changes_assignment(self) -> None:
        groups = _uniform_groups(12)
        a = make_group_split_map(
            groups, GroupSplitConfig(val_ratio=0.25, test_ratio=0.25, seed=1)
        )
        b = make_group_split_map(
            groups, GroupSplitConfig(val_ratio=0.25, test_ratio=0.25, seed=999)
        )
        # Same number of holdout groups, but membership differs across seeds.
        assert a != b


class TestGroupSplitConfig:
    def test_is_frozen(self) -> None:
        config = GroupSplitConfig(val_ratio=0.1, test_ratio=0.1, seed=0)
        with pytest.raises(FrozenInstanceError):
            config.seed = 5  # type: ignore[misc]
