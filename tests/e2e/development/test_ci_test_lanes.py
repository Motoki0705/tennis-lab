"""The actual Actions matrix must partition every eligible test exactly once."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.automation.ci.sharding import CI_EXCLUDED_FILES, discover_test_files

REPO_ROOT = Path(__file__).resolve().parents[3]


def _list_tests(*args: str) -> tuple[str, ...]:
    completed = subprocess.run(
        [sys.executable, "-m", "spin", "ci", *args, "--list-tests"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(completed.stdout.splitlines())


def test_actions_matrix_covers_every_ci_test_once_and_excludes_manual_integration() -> (
    None
):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    matrix = workflow["jobs"]["lint-and-test"]["strategy"]["matrix"]["shard"]
    assert matrix == list(range(1, len(matrix) + 1))
    groups = [
        _list_tests("--shard", str(index), "--shards", str(len(matrix)))
        for index in matrix
    ]
    counts = Counter(path for group in groups for path in group)
    expected = set(discover_test_files(REPO_ROOT)) - CI_EXCLUDED_FILES.keys()
    assert set(counts) == expected
    assert set(counts.values()) == {1}
    assert set(_list_tests()) == expected
    for group in groups:
        assert group
        assert group == tuple(sorted(group))
    assert set(CI_EXCLUDED_FILES) == {
        "tests/integration/synthetic_data_generation/test_court_dataset.py",
        "tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py",
    }


@pytest.mark.parametrize(
    "args", [("--shard", "1"), ("--shards", "4"), ("--shard", "5", "--shards", "4")]
)
def test_invalid_shard_arguments_fail_before_running_tests(
    args: tuple[str, ...],
) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "spin", "ci", *args, "--list-tests"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Error:" in completed.stderr


def test_cpu_dependency_group_preserves_default_gpu_environment() -> None:
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    assert "gpu" in config["tool"]["uv"]["default-groups"]
    assert [{"group": "cpu"}, {"group": "gpu"}] in config["tool"]["uv"]["conflicts"]
    for name in ("torch", "torchvision"):
        registries = {
            package["source"]["registry"]
            for package in lock["package"]
            if package["name"] == name
        }
        assert registries == {
            "https://pypi.org/simple",
            "https://download.pytorch.org/whl/cpu",
        }
