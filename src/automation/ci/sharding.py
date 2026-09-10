"""Partition discovered test files using a versioned timing profile."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CI_EXCLUDED_FILES = {
    "tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py": "Long scene generation/recovery integration; run explicitly with spin test.",
}
DEFAULT_FILE_SECONDS = 1.0


def validate_test_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not value.startswith("tests/")
        or path.as_posix() != value
        or not (path.name.startswith("test_") or path.name.endswith("_test.py"))
        or path.suffix != ".py"
    ):
        raise ValueError(f"Invalid repository-relative test path: {value!r}")
    return value


def discover_test_files(repo_root: Path) -> tuple[str, ...]:
    tests_root = repo_root / "tests"
    if not tests_root.is_dir():
        raise FileNotFoundError(f"Tests directory is unavailable: {tests_root}")
    files = tuple(
        sorted(
            path.relative_to(repo_root).as_posix()
            for path in tests_root.rglob("*.py")
            if path.name.startswith("test_") or path.name.endswith("_test.py")
        )
    )
    if not files:
        raise ValueError("No test files were discovered")
    return files


def ci_test_files(repo_root: Path) -> tuple[str, ...]:
    files = discover_test_files(repo_root)
    missing = CI_EXCLUDED_FILES.keys() - set(files)
    if missing:
        raise ValueError(f"Stale CI exclusions: {sorted(missing)}")
    return tuple(path for path in files if path not in CI_EXCLUDED_FILES)


def read_durations(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported timing profile: {path}")
    values = payload.get("seconds_by_file")
    if not isinstance(values, dict):
        raise ValueError(f"Timing profile needs seconds_by_file: {path}")
    result: dict[str, float] = {}
    for name, value in values.items():
        validate_test_path(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid duration for {name}: {value!r}")
        result[name] = float(value)
    return result


@dataclass(frozen=True)
class Shard:
    index: int
    count: int
    files: tuple[str, ...]
    estimated_seconds: float
    unmeasured_files: tuple[str, ...]


def partition_tests(
    files: Sequence[str], durations: Mapping[str, float], *, count: int
) -> tuple[Shard, ...]:
    """Longest-first placement, with stable ordering and no dropped new files."""
    if len(set(files)) != len(files):
        raise ValueError("Duplicate test files in partition input")
    if not 1 <= count <= len(files):
        raise ValueError("Shard count must be between 1 and the number of test files")
    for path in files:
        validate_test_path(path)
    weights = {path: durations.get(path, DEFAULT_FILE_SECONDS) for path in files}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Durations must be finite and nonnegative")
    groups: list[list[str]] = [[] for _ in range(count)]
    totals = [0.0] * count
    for path in sorted(files, key=lambda name: (-weights[name], name)):
        index = min(range(count), key=lambda i: (totals[i], len(groups[i]), i))
        groups[index].append(path)
        totals[index] += weights[path]
    return tuple(
        Shard(
            index=index + 1,
            count=count,
            files=tuple(sorted(group)),
            estimated_seconds=totals[index],
            unmeasured_files=tuple(
                sorted(path for path in group if path not in durations)
            ),
        )
        for index, group in enumerate(groups)
    )
