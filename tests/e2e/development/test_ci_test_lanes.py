"""Actions must preserve CPU test coverage and propagate pytest failures."""

from __future__ import annotations

import json
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
    assert len(commands) == 1
    assert commands[0] == [".venv/bin/python", "-m", "spin", "lint"]
    command = shlex.split(
        next(step["run"] for step in job["steps"] if step["name"] == "Test")
    )
    assert command[:5] == [
        ".venv/bin/python",
        "-m",
        "scripts.ci_metrics",
        "run",
        "--output",
    ]
    assert command[-2:] == ["--workers", "2"]
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
    shutil.copytree(
        REPO_ROOT / "scripts/ci_metrics",
        tmp_path / "scripts/ci_metrics",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src/__init__.py").touch()
    excluded_files = {
        "tests/integration/synthetic_data_generation/test_court_dataset.py",
    }
    assert all((REPO_ROOT / path).is_file() for path in excluded_files)
    samples = {
        "tests/unit/test_unit.py": "def test_unit(): pass\n",
        "tests/integration/test_integration.py": "def test_integration(): pass\n",
        "tests/e2e/test_e2e.py": "def test_e2e(): pass\n",
        "tests/newly_added_test.py": f"def test_new(): assert {not failing_test}\n",
        "tests/test_setup_outcomes.py": (
            "import pytest\n"
            "@pytest.fixture\n"
            "def skipped_fixture(): pytest.skip('setup skip')\n"
            "def test_setup_skip(skipped_fixture): raise AssertionError('unreachable')\n"
            "@pytest.fixture\n"
            "def broken_fixture(): raise RuntimeError('setup failure')\n"
            + (
                "def test_setup_error(broken_fixture): raise AssertionError('unreachable')\n"
                if failing_test
                else ""
            )
        ),
        "tests/test_collect_skip.py": "import pytest\npytest.skip('collection skip', allow_module_level=True)\n",
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
        env={
            **os.environ,
            **job["env"],
            "PYTEST_LOCK": str(tmp_path / "inner-pytest.lock"),
        },
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == int(failing_test), output
    measured = tmp_path / ".ci-metrics/tests"
    summary = json.loads((measured / "summary.json").read_text())
    inventory = json.loads((measured / "inventory.json").read_text())
    assert summary["selected_count"] == (6 if failing_test else 5)
    assert summary["successful_test_baseline"] is not failing_test
    assert summary["issues"] == []
    assert summary["counts"]["skipped"] == 1
    skipped = next(
        item for item in summary["fixtures"] if "skipped_fixture" in item["fixture"]
    )
    assert skipped["errors"] == 0
    assert skipped["skipped"] == 1
    if failing_test:
        assert summary["counts"]["failed"] == 2
        broken = next(
            item for item in summary["fixtures"] if "broken_fixture" in item["fixture"]
        )
        assert broken["errors"] == 1
        results = json.loads((measured / "results.json").read_text())
        setup_error = next(
            row for row in results if row["nodeid"].endswith("::test_setup_error")
        )
        assert "setup failure" in setup_error["failure_details"]["setup"]
        call_error = next(
            row for row in results if row["nodeid"].endswith("::test_new")
        )
        assert "AssertionError" in call_error["failure_details"]["call"]
    assert {nodeid.split("::")[0] for nodeid in inventory} == {
        "tests/unit/test_unit.py",
        "tests/integration/test_integration.py",
        "tests/e2e/test_e2e.py",
        "tests/newly_added_test.py",
        "tests/test_setup_outcomes.py",
    }
    log = (measured / "pytest.log").read_text()
    assert f"2 workers [{6 if failing_test else 5} items]" in log
    assert ("1 failed, 3 passed" if failing_test else "4 passed") in log


def test_ci_measurements_are_retained_when_preparation_or_tests_fail() -> None:
    steps = _ci_job()["steps"]
    by_name = {step["name"]: step for step in steps}
    assert by_name["Summarize test measurements"]["if"] == "always()"
    assert by_name["Snapshot attempt timings"]["if"] == "always()"
    assert by_name["Upload test measurements"]["if"] == "always()"
    assert by_name["Upload test measurements"]["with"]["if-no-files-found"] == "error"
    assert [step["name"] for step in steps].index("Record runner environment") < [
        step["name"] for step in steps
    ].index("Install locked CPU environment")


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
