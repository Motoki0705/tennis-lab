"""Coverage and stability invariants for duration-based file partitioning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.automation.ci.sharding import (
    CI_EXCLUDED_FILES,
    ci_test_files,
    partition_tests,
    read_durations,
)


def test_longest_files_are_balanced_and_new_files_are_never_lost() -> None:
    files = [f"tests/test_{letter}.py" for letter in "abcdef"]
    durations = dict(zip(files[:4], (10.0, 9.0, 2.0, 1.0), strict=True))
    groups = partition_tests(files, durations, count=2)
    assert [group.estimated_seconds for group in groups] == [12.0, 12.0]
    assert set(groups[0].files).isdisjoint(groups[1].files)
    assert set(groups[0].files + groups[1].files) == set(files)
    assert set(groups[0].unmeasured_files + groups[1].unmeasured_files) == set(
        files[4:]
    )
    assert partition_tests(list(reversed(files)), durations, count=2) == groups


def test_zero_duration_files_still_fill_every_shard() -> None:
    files = [f"tests/test_{index}.py" for index in range(8)]
    groups = partition_tests(files, dict.fromkeys(files, 0.0), count=4)
    assert [len(group.files) for group in groups] == [2, 2, 2, 2]


@pytest.mark.parametrize("count", [0, -1, 3])
def test_invalid_shard_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="Shard count"):
        partition_tests(["tests/test_a.py", "tests/test_b.py"], {}, count=count)


def test_duplicate_files_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        partition_tests(["tests/test_a.py", "tests/test_a.py"], {}, count=1)


@pytest.mark.parametrize("value", [-1, "5", True, float("nan"), float("inf")])
def test_invalid_timing_values_do_not_silently_change_the_plan(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps({"schema_version": 1, "seconds_by_file": {"tests/test_a.py": value}})
    )
    with pytest.raises(ValueError, match="Invalid duration"):
        read_durations(path)


def test_missing_and_malformed_profiles_fail(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    with pytest.raises(FileNotFoundError):
        read_durations(path)
    path.write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="Unsupported"):
        read_durations(path)


def test_discovery_excludes_only_the_explicit_manual_test(tmp_path: Path) -> None:
    excluded = tmp_path / next(iter(CI_EXCLUDED_FILES))
    excluded.parent.mkdir(parents=True)
    excluded.touch()
    normal = tmp_path / "tests/test_new.py"
    normal.touch()
    assert ci_test_files(tmp_path) == ("tests/test_new.py",)
    excluded.unlink()
    with pytest.raises(ValueError, match="Stale CI exclusions"):
        ci_test_files(tmp_path)
