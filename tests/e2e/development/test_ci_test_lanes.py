"""Actions must preserve CPU test coverage and propagate pytest failures."""

from __future__ import annotations

import importlib
import json
import os
import re
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
    assert set(workflow["jobs"]) == {"lint", "tests", "python"}
    return cast(dict[str, Any], workflow["jobs"]["tests"])


def test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs() -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"] is None
    assert "paths" not in triggers["push"] and "paths-ignore" not in triggers["push"]
    assert workflow["concurrency"] == {
        "group": "ci-${{ github.event.pull_request.number || github.ref }}",
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_actions_parallelize_lint_and_cpu_shards_and_always_aggregate() -> None:
    job = _ci_job()
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    choices = triggers["workflow_dispatch"]["inputs"]["shards"]
    default = choices["default"]
    shards = list(range(int(default)))
    expression = job["strategy"]["matrix"]["shard"]
    alternatives = dict(
        re.findall(r"inputs.shards == '(\d+)' && '(\[[\d,]+\])'", expression)
    )
    fallback = re.search(r"\|\| '(\[[\d,]+\])'\) }}$", expression)
    assert fallback is not None
    fallback_json = fallback[1]
    assert isinstance(fallback_json, str)
    assert json.loads(fallback_json) == shards
    assert set(alternatives) == set(choices["options"]) - {default}
    for count in choices["options"]:
        assert json.loads(alternatives.get(count, fallback_json)) == list(
            range(int(count))
        )
    assert job["strategy"]["fail-fast"] is False
    assert "needs" not in job
    assert job["timeout-minutes"] == "${{ inputs.shards == '1' && 120 || 60 }}"
    assert job["env"]["PYTEST_XDIST_AUTO_NUM_WORKERS"] == "${{ inputs.workers || '2' }}"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert job["env"][variable] == "1"

    assert choices["options"] == ["1", "2", "4", "6", "8", "10"]
    assert choices["default"] == "10"
    assert workflow["env"]["CI_SHARD_COUNT"] == "${{ inputs.shards || '10' }}"
    workers = triggers["workflow_dispatch"]["inputs"]["workers"]
    assert workers["options"] == ["2", "4"]
    assert workers["default"] == "2"
    assert workflow["env"]["CI_WORKERS"] == "${{ inputs.workers || '2' }}"
    lint = workflow["jobs"]["lint"]
    assert "needs" not in lint
    commands = [
        shlex.split(step["run"])
        for step in lint["steps"]
        if step.get("run", "").startswith(".venv/bin/python -m spin ")
    ]
    assert len(commands) == 1
    assert commands[0] == [".venv/bin/python", "-m", "spin", "lint"]
    command = shlex.split(
        next(step["run"] for step in job["steps"] if step["name"] == "Test")
        .replace("${{ matrix.shard }}", "0")
        .replace("$CI_SHARD_COUNT", default)
        .replace("$CI_WORKERS", "2")
    )
    assert command[:5] == [
        ".venv/bin/python",
        "-m",
        "scripts.ci_metrics",
        "run",
        "--output",
    ]
    assert command[command.index("--workers") + 1] == "2"
    assert command[command.index("--shard-count") + 1] == str(len(shards))
    aggregate = workflow["jobs"]["python"]
    assert aggregate["name"] == "Python"
    assert aggregate["if"] == "always()"
    assert set(aggregate["needs"]) == {"lint", "tests"}
    verification = next(
        step
        for step in aggregate["steps"]
        if step["name"] == "Verify jobs and complete CPU coverage"
    )
    assert verification["if"] == "always()"
    assert verification["env"]["CI_JOB_RESULTS"] == "${{ toJSON(needs) }}"
    assert shlex.split(verification["run"].replace("$CI_SHARD_COUNT", default))[
        -2:
    ] == [
        "--count",
        str(len(shards)),
    ]
    assert aggregate["steps"][-1]["if"] == "always()"
    assert aggregate["steps"][-1]["with"]["if-no-files-found"] == "error"
    assert "uv sync --locked --no-group gpu --group cpu" in [
        step.get("run") for step in job["steps"]
    ]


@pytest.mark.parametrize("failing_test", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("shard_count", [1, 2, 4, 6, 8, 10])
def test_actions_test_command_discovers_tests_preserves_exclusions_and_exit_code(
    tmp_path: Path,
    failing_test: bool,
    shard_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _ci_job()
    template = next(step["run"] for step in job["steps"] if step["name"] == "Test")
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
        **{f"tests/test_extra_{i}.py": "def test_extra(): pass\n" for i in range(5)},
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

    profile = tmp_path / "scripts/ci_metrics/cpu-file-times.json"
    profile.write_text(
        json.dumps(
            {"schema_version": 1, "file_seconds": {"tests/unit/test_unit.py": 1}}
        )
    )
    inventory = []
    summaries = []
    results = []
    failures = 0
    signatures = []
    job_logs = []
    monkeypatch.syspath_prepend(str(REPO_ROOT / ".agents/ops"))
    classify = importlib.import_module("ci_watch.classify")
    for index in range(shard_count):
        command = shlex.split(
            template.replace("${{ matrix.shard }}", str(index))
            .replace("$CI_SHARD_COUNT", str(shard_count))
            .replace("$CI_WORKERS", "4" if shard_count == 4 else "2")
        )
        completed = subprocess.run(
            [sys.executable, *command[1:]],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                **job["env"],
                "PYTEST_XDIST_AUTO_NUM_WORKERS": "4" if shard_count == 4 else "2",
            },
            timeout=60,
        )
        output = completed.stdout + completed.stderr
        job_logs.append(output)
        assert completed.returncode in {0, 1}, output
        if completed.returncode:
            signatures.extend(
                classify.signatures(
                    "ci.yml",
                    {
                        "name": f"CPU tests ({index})",
                        "conclusion": "failure",
                        "steps": [{"name": "Test", "conclusion": "failure"}],
                    },
                    output,
                    {"endpoint": f"fixture job log {index}", "object": "fixture"},
                )
            )
        measured = tmp_path / f".ci-metrics/shard-{index}/tests"
        assert (measured / "pytest.log").read_text() in output
        summary = json.loads((measured / "summary.json").read_text())
        assert summary["issues"] == []
        assert summary["successful_test_baseline"] == (completed.returncode == 0)
        assert summary["full_cpu_baseline"] is False
        failures += summary["counts"].get("failed", 0)
        inventory.extend(json.loads((measured / "inventory.json").read_text()))
        results.extend(json.loads((measured / "results.json").read_text()))
        summaries.append(summary)
    assert len(inventory) == len(set(inventory)) == (11 if failing_test else 10)
    assert failures == (2 if failing_test else 0)
    assert sum(s["counts"].get("skipped", 0) for s in summaries) == 1
    skipped = next(
        item
        for s in summaries
        for item in s["fixtures"]
        if "skipped_fixture" in item["fixture"]
    )
    assert skipped["errors"] == 0
    assert skipped["skipped"] == 1
    if failing_test:
        broken = next(
            item
            for s in summaries
            for item in s["fixtures"]
            if "broken_fixture" in item["fixture"]
        )
        assert broken["errors"] == 1
        setup_error = next(
            row for row in results if row["nodeid"].endswith("::test_setup_error")
        )
        assert "setup failure" in setup_error["failure_details"]["setup"]
        call_error = next(
            row for row in results if row["nodeid"].endswith("::test_new")
        )
        assert "AssertionError" in call_error["failure_details"]["call"]
        # Failure identities survive each real matrix job log.
        assert {row["test_id"] for row in signatures} == {
            "tests/newly_added_test.py::test_new",
            "tests/test_setup_outcomes.py::test_setup_error",
        }
        assert all(row["specific"] for row in signatures)
        assert len({row["signature"] for row in signatures}) == 2
        combined_log = "\n".join(job_logs)
        assert (
            "AssertionError" in combined_log
            and "RuntimeError: setup failure" in combined_log
        )
    assert {nodeid.split("::")[0] for nodeid in inventory} == {
        "tests/unit/test_unit.py",
        "tests/integration/test_integration.py",
        "tests/e2e/test_e2e.py",
        "tests/newly_added_test.py",
        "tests/test_setup_outcomes.py",
        *(f"tests/test_extra_{i}.py" for i in range(5)),
    }


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


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("existing", 0),
        ("install", 0),
        ("update_failure", 42),
        ("install_failure", 43),
        ("timeout", 124),
        ("missing_after", 127),
        ("version_failure", 44),
    ],
)
def test_system_dependencies_are_verified_and_failures_are_not_hidden(
    tmp_path: Path, scenario: str, expected: int
) -> None:
    binary = tmp_path / "bin"
    binary.mkdir()
    log = tmp_path / "calls.jsonl"

    def executable(name: str, source: str) -> None:
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)

    executable("ffmpeg", "print('ffmpeg version fixture')\n")
    if scenario in {"existing", "version_failure"}:
        executable(
            "rclone",
            "print('rclone fixture'); raise SystemExit("
            + ("44" if scenario == "version_failure" else "0")
            + ")\n",
        )
    common = (
        "import json, os, sys\nfrom pathlib import Path\n"
        f"scenario = {scenario!r}\n"
        f"with open({str(log)!r}, 'a') as stream: "
        "stream.write(json.dumps(sys.argv) + '\\n')\n"
    )
    executable(
        "timeout",
        common
        + "assert sys.argv[1] == '180s'\n"
        + "if scenario == 'timeout': raise SystemExit(124)\n"
        + "os.execvp(sys.argv[2], sys.argv[2:])\n",
    )
    installed_script = f"#!{sys.executable}\nprint(chr(82))\n"
    executable(
        "sudo",
        common
        + "assert sys.argv[1] == 'apt-get'\n"
        + "if 'update' in sys.argv and scenario == 'update_failure': raise SystemExit(42)\n"
        + "if 'install' in sys.argv:\n"
        + "    assert sys.argv[-1] == 'rclone' and 'ffmpeg' not in sys.argv\n"
        + "    if scenario == 'install_failure': raise SystemExit(43)\n"
        + "    if scenario != 'missing_after':\n"
        + f"        path = Path({str(binary / 'rclone')!r})\n"
        + f"        path.write_text({installed_script!r})\n"
        + "        path.chmod(0o755)\n",
    )
    step = next(
        s for s in _ci_job()["steps"] if s["name"] == "Install system dependencies"
    )
    assert step["timeout-minutes"] == 8
    completed = subprocess.run(
        ["/bin/bash", *shlex.split(step["run"])[1:]],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": str(binary)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    assert bool(calls) == (scenario not in {"existing", "version_failure"})
    if expected == 0:
        assert "ffmpeg version fixture" in completed.stdout
    if scenario == "update_failure":
        assert not any("install" in row for row in calls)


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
