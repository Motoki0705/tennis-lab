"""End-to-end tests for the spin CI test-lane selector."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"
EXPECTED_LONG_TAIL = {
    "tests/e2e/development/test_configuration_audit.py",
    "tests/unit/synthetic_data_generation/alignment/test_evidence_source.py",
    "tests/unit/tasks/plcs/test_configuration_contracts.py",
    "tests/unit/utils/configuration/test_discovery.py",
    "tests/unit/utils/configuration/test_inventory.py",
}
EXPECTED_SCENE_PIPELINE = {
    "tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py"
}


def _list_lane_tests(lane: str) -> tuple[str, ...]:
    completed = subprocess.run(
        [sys.executable, "-m", "spin", "ci", "--lane", lane, "--list-tests"],
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
    long_tail = set(_list_lane_tests("long-tail"))
    scene_pipeline = set(_list_lane_tests("scene-pipeline"))
    remainder = set(_list_lane_tests("remainder"))
    specialized = long_tail | scene_pipeline

    assert long_tail == EXPECTED_LONG_TAIL
    assert scene_pipeline == EXPECTED_SCENE_PIPELINE
    assert long_tail.isdisjoint(scene_pipeline)
    assert specialized.isdisjoint(remainder)
    assert specialized | remainder == _discover_test_files()


def test_ci_lane_output_is_sorted_unique_and_repo_relative() -> None:
    for lane in ("long-tail", "remainder", "scene-pipeline"):
        selected = _list_lane_tests(lane)

        assert selected
        assert selected == tuple(sorted(selected))
        assert len(selected) == len(set(selected))
        assert all(not Path(path).is_absolute() for path in selected)
        assert all((REPO_ROOT / path).is_file() for path in selected)


def test_github_actions_delegates_repository_checks_to_spin() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert 'uv run --locked spin ci --lane "${{ matrix.lane }}"' in workflow
    assert "scripts/ci/select_test_lane.py" not in workflow
    assert "python -m pytest" not in workflow
