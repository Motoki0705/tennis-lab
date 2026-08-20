"""Select complete, non-overlapping pytest file lanes for GitHub Actions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANES = ("long-tail", "remainder", "scene-pipeline")
LONG_TAIL_TEST_FILES = frozenset(
    {
        "tests/e2e/development/test_configuration_audit.py",
        "tests/unit/synthetic_data_generation/alignment/test_evidence_source.py",
        "tests/unit/tasks/plcs/test_configuration_contracts.py",
        "tests/unit/utils/configuration/test_discovery.py",
        "tests/unit/utils/configuration/test_inventory.py",
    }
)
SCENE_PIPELINE_TEST_FILES = frozenset(
    {"tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py"}
)
SPECIALIZED_TEST_FILES = LONG_TAIL_TEST_FILES | SCENE_PIPELINE_TEST_FILES


def discover_test_files(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return every pytest file using repository-relative POSIX paths."""

    tests_root = repo_root / "tests"
    if not tests_root.is_dir():
        raise FileNotFoundError(f"Tests directory is unavailable: {tests_root}")

    test_files = tuple(
        sorted(
            path.relative_to(repo_root).as_posix()
            for path in tests_root.rglob("*.py")
            if path.name.startswith("test_") or path.name.endswith("_test.py")
        )
    )
    if not test_files:
        raise RuntimeError(f"No pytest files were found under {tests_root}")
    return test_files


def select_test_files(
    lane: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, ...]:
    """Return exactly the test files assigned to one CI lane."""

    if lane not in LANES:
        choices = ", ".join(LANES)
        raise ValueError(f"Unknown CI test lane {lane!r}; expected one of: {choices}")

    all_files = frozenset(discover_test_files(repo_root))
    missing = sorted(SPECIALIZED_TEST_FILES - all_files)
    if missing:
        rendered = ", ".join(missing)
        raise FileNotFoundError(
            f"Configured specialized test files are unavailable: {rendered}"
        )

    if lane == "long-tail":
        selected = LONG_TAIL_TEST_FILES
    elif lane == "scene-pipeline":
        selected = SCENE_PIPELINE_TEST_FILES
    else:
        selected = all_files - SPECIALIZED_TEST_FILES

    if not selected:
        raise RuntimeError(f"CI test lane {lane!r} is empty")
    return tuple(sorted(selected))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the repository-relative pytest files for one CI lane."
    )
    parser.add_argument("lane", choices=LANES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for test_file in select_test_files(args.lane):
        print(test_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
