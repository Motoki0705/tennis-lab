"""End-to-end tests for the CI test-lane selector."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTOR = REPO_ROOT / "scripts/ci/select_test_lane.py"
EXPECTED_LONG_TAIL = {
    "tests/e2e/development/test_configuration_audit.py",
    "tests/unit/synthetic_data_generation/alignment/test_evidence_source.py",
    "tests/unit/tasks/plcs/test_configuration_contracts.py",
    "tests/unit/utils/configuration/test_discovery.py",
    "tests/unit/utils/configuration/test_inventory.py",
}


def _run_selector(lane: str) -> tuple[str, ...]:
    completed = subprocess.run(
        [sys.executable, str(SELECTOR), lane],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(completed.stdout.splitlines())


def _discover_test_files() -> set[str]:
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
    }


def test_ci_lanes_partition_every_test_file_exactly_once() -> None:
    long_tail = set(_run_selector("long-tail"))
    remainder = set(_run_selector("remainder"))

    assert long_tail == EXPECTED_LONG_TAIL
    assert long_tail.isdisjoint(remainder)
    assert long_tail | remainder == _discover_test_files()


def test_ci_lane_output_is_sorted_unique_and_repo_relative() -> None:
    for lane in ("long-tail", "remainder"):
        selected = _run_selector(lane)

        assert selected
        assert selected == tuple(sorted(selected))
        assert len(selected) == len(set(selected))
        assert all(not Path(path).is_absolute() for path in selected)
        assert all((REPO_ROOT / path).is_file() for path in selected)
