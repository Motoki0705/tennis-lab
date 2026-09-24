"""Actions must preserve CPU test coverage and propagate pytest failures."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ci_job() -> dict[str, Any]:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert set(workflow["jobs"]) == {"lint-and-test"}
    return cast(dict[str, Any], workflow["jobs"]["lint-and-test"])


def test_actions_use_one_cpu_job_with_bounded_parallelism() -> None:
    job = _ci_job()
    assert "strategy" not in job
    assert job["timeout-minutes"] == 60
    assert job["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "2"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert job["env"][variable] == "1"

    commands = [
        shlex.split(step["run"])
        for step in job["steps"]
        if step.get("run", "").startswith(".venv/bin/python -m spin ")
    ]
    assert len(commands) == 2
    assert commands[0] == [".venv/bin/python", "-m", "spin", "lint"]
    assert commands[1][:4] == [".venv/bin/python", "-m", "spin", "test"]
    assert "uv sync --locked --no-group gpu --group cpu" in [
        step.get("run") for step in job["steps"]
    ]


@pytest.mark.parametrize("failing_test", [False, True], ids=["success", "failure"])
def test_actions_test_command_discovers_tests_preserves_exclusions_and_exit_code(
    tmp_path: Path, failing_test: bool
) -> None:
    job = _ci_job()
    command = shlex.split(
        next(step["run"] for step in job["steps"] if step["name"] == "Test")
    )
    # Use the repository CLI and pytest settings with a small independent suite.
    (tmp_path / ".spin").mkdir()
    shutil.copy2(REPO_ROOT / ".spin/cmds.py", tmp_path / ".spin/cmds.py")
    shutil.copy2(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").touch()
    excluded_files = {
        "tests/integration/synthetic_data_generation/test_scene_pipeline_cpu.py",
        "tests/integration/synthetic_data_generation/test_court_dataset.py",
    }
    assert {
        arg.removeprefix("--ignore=")
        for arg in command
        if arg.startswith("--ignore=")
    } == excluded_files
    assert all((REPO_ROOT / path).is_file() for path in excluded_files)
    samples = {
        "tests/unit/test_unit.py": "def test_unit(): pass\n",
        "tests/integration/test_integration.py": "def test_integration(): pass\n",
        "tests/e2e/test_e2e.py": "def test_e2e(): pass\n",
        "tests/newly_added_test.py": f"def test_new(): assert {not failing_test}\n",
        "tests/test_environmental.py": (
            "import pytest\n"
            "@pytest.mark.local_data\n"
            "def test_local_data(): raise AssertionError('local data must be excluded')\n"
            "@pytest.mark.cuda\n"
            "def test_cuda(): raise AssertionError('CUDA must be excluded')\n"
        ),
        **dict.fromkeys(
            excluded_files,
            "raise AssertionError('manual integration must not be collected')\n",
        ),
    }
    for relative, source in samples.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, *command[1:]],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **job["env"]},
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == int(failing_test), output
    assert "2 workers [4 items]" in output
    assert ("1 failed, 3 passed" if failing_test else "4 passed") in output


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
