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

from scripts.ci_metrics import plan as lane_planner
from scripts.ci_metrics.lanes import (
    REGISTRY,
    load_registry,
    make_plan,
    partition_nodeids,
    validate_plan,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ci_job() -> dict[str, Any]:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert set(workflow["jobs"]) == {"plan", "lint", "tests", "python"}
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
    assert job["needs"] == ["plan"]
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
    assert set(aggregate["needs"]) == {"plan", "lint", "tests"}
    planning = workflow["jobs"]["plan"]
    assert "needs" not in planning
    assert not any("uv sync" in step.get("run", "") for step in planning["steps"])
    plan_command = next(
        step["run"] for step in planning["steps"] if step["name"] == "Plan CPU lanes"
    )
    assert shlex.split(plan_command)[:4] == [
        "python3",
        "-m",
        "scripts.ci_metrics.plan",
        "--output",
    ]
    assert {"push", "pull_request", "workflow_dispatch"} == set(triggers)
    assert command[command.index("--lane-plan") + 1] == ".ci-plan/plan.json"
    for consumer in (job, aggregate):
        download = next(
            step
            for step in consumer["steps"]
            if step["name"] == "Download CPU lane plan"
        )
        assert download["with"]["name"] == planning["steps"][-1]["with"]["name"]
    verification = next(
        step
        for step in aggregate["steps"]
        if step["name"] == "Verify jobs and complete CPU coverage"
    )
    assert verification["if"] == "always()"
    assert verification["env"]["CI_JOB_RESULTS"] == "${{ toJSON(needs) }}"
    verification_command = shlex.split(
        verification["run"].replace("$CI_SHARD_COUNT", default)
    )
    assert verification_command[verification_command.index("--count") + 1] == str(
        len(shards)
    )
    assert (
        verification_command[verification_command.index("--lane-plan") + 1]
        == ".ci-plan/plan.json"
    )
    assert aggregate["steps"][-1]["if"] == "always()"
    assert aggregate["steps"][-1]["with"]["if-no-files-found"] == "error"
    assert "uv sync --locked --no-group gpu --group cpu" in [
        step.get("run") for step in job["steps"]
    ]


@pytest.mark.parametrize(
    "failing_test,shard_count,workers",
    [(False, 1, 1), (True, 2, 2)],
    ids=["success", "failure"],
)
def test_actions_test_command_discovers_tests_preserves_exclusions_and_exit_code(
    tmp_path: Path,
    failing_test: bool,
    shard_count: int,
    workers: int,
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
    (tmp_path / REGISTRY).write_text(json.dumps({"schema_version": 1, "suites": []}))
    registry, files = load_registry(tmp_path)
    lane_plan = make_plan(
        registry,
        files,
        source_sha=os.environ.get("GITHUB_SHA"),
        event="local",
        run_id=os.environ.get("GITHUB_RUN_ID"),
        attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
    )
    (tmp_path / ".ci-plan").mkdir()
    (tmp_path / ".ci-plan/plan.json").write_text(json.dumps(lane_plan))

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
            .replace("$CI_WORKERS", str(workers))
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
                "PYTEST_XDIST_AUTO_NUM_WORKERS": str(workers),
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


@pytest.fixture
def lane_registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suites": [
            {
                "id": "wide",
                "files": ["tests/test_wide.py"],
                "paths": ["tools/wide/**", "tests/wide/**"],
                "reason": "Independent broad runner regression",
                "owner": "repository-owner",
            }
        ],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate_file",
        "duplicate_id",
        "missing_file",
        "glob_file",
        "missing_reason",
        "blank_reason",
        "missing_owner",
        "blank_owner",
        "empty_paths",
        "absolute_path",
        "parent_path",
        "unknown_field",
    ],
)
def test_lane_registry_rejects_invalid_exceptions(
    lane_registry: dict[str, Any], fault: str
) -> None:
    import copy

    suite = lane_registry["suites"][0]
    if fault in {"duplicate_file", "duplicate_id"}:
        extra = copy.deepcopy(suite)
        if fault == "duplicate_file":
            extra["id"] = "second"
        lane_registry["suites"].append(extra)
    elif fault in {"missing_file", "glob_file"}:
        suite["files"] = [
            "tests/missing.py" if fault == "missing_file" else "tests/test_*.py"
        ]
    elif fault.startswith("missing_"):
        del suite[fault.removeprefix("missing_")]
    elif fault.startswith("blank_"):
        suite[fault.removeprefix("blank_")] = "  "
    elif fault == "empty_paths":
        suite["paths"] = []
    elif fault == "absolute_path":
        suite["paths"] = ["/tools/**"]
    elif fault == "parent_path":
        suite["paths"] = ["tools/../**"]
    else:
        suite["weekly"] = True
    with pytest.raises(ValueError):
        validate_registry(lane_registry, ["tests/test_wide.py"])


@pytest.mark.parametrize(
    "path,status,scope",
    [
        ("docs/readme.md", "M", "pr"),
        ("tools/wide/runner.py", "M", "pr"),
        ("tests/test_wide.py", "M", "pr"),
        ("src/utils/math.py", "M", "full"),
        ("src/tasks/base/model.py", "M", "full"),
        ("tests/support/factory.py", "M", "full"),
        ("tests/wide/conftest.py", "M", "full"),
        ("conftest.py", "M", "full"),
        ("uv.lock", "M", "full"),
        ("pyproject.toml", "M", "full"),
        (".spin/cmds.py", "M", "full"),
        (".github/workflows/ci.yml", "M", "full"),
        (REGISTRY, "M", "full"),
        ("tools/unrelated.py", "D", "full"),
        ("old.py", "R100", "full"),
    ],
)
def test_pr_plan_keeps_unregistered_tests_and_promotes_shared_or_uncertain_changes(
    lane_registry: dict[str, Any],
    path: str,
    status: str,
    scope: str,
) -> None:
    files = ["tests/test_new.py", "tests/test_wide.py"]
    changes = lane_planner.parse_diff(
        (
            f"{status}\0{path}\0" + ("renamed.py\0" if status.startswith("R") else "")
        ).encode()
    )
    plan = validate_plan(
        make_plan(
            lane_registry,
            files,
            source_sha="head",
            event="pull_request",
            base_sha="base",
            head_sha="head",
            changes=changes,
        )
    )
    assert plan["scope"] == scope
    requested, deferred = partition_nodeids(
        [
            "tests/test_new.py::test_new",
            "tests/test_wide.py::test_existing",
            "tests/test_wide.py::test_added",
        ],
        plan,
    )
    assert "tests/test_new.py::test_new" in requested
    should_run_wide = scope == "full" or path in {
        "tools/wide/runner.py",
        "tests/test_wide.py",
    }
    assert ("tests/test_wide.py::test_existing" in requested) == should_run_wide
    assert len(deferred) == (0 if should_run_wide else 2)
    if scope == "full":
        assert any(
            reason.startswith(("shared_foundation:", "rename_or_deletion:"))
            for reason in plan["reasons"]
        )


@pytest.mark.parametrize(
    "event,ref",
    [
        ("pull_request", "refs/pull/1/merge"),
        ("push", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
    ],
)
def test_initial_empty_registry_preserves_the_current_entire_cpu_target(
    event: str, ref: str
) -> None:
    load_registry(REPO_ROOT)
    files = ["tests/test_one.py", "tests/test_two.py"]
    universe = [f"{f}::test_{i}" for f in files for i in range(3)]
    plan = validate_plan(
        make_plan(
            {"schema_version": 1, "suites": []},
            files,
            source_sha="head",
            event=event,
            ref=ref,
            base_sha="base",
            head_sha="head",
        )
    )
    assert partition_nodeids(universe, plan) == (universe, {})
    if event != "pull_request":
        assert plan["scope"] == "full"
        assert plan["reasons"] == ["main_push" if event == "push" else "manual_full"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_main_and_manual_full_request_registered_suites_without_path_changes(
    lane_registry: dict[str, Any], event: str
) -> None:
    universe = ["tests/test_wide.py::test_one"]
    plan = validate_plan(
        make_plan(
            lane_registry,
            ["tests/test_wide.py"],
            source_sha="head",
            event=event,
            ref="refs/heads/main",
        )
    )
    assert plan["scope"] == "full"
    assert plan["requested_suites"] == ["wide"]
    assert partition_nodeids(universe, plan) == (universe, {})


@pytest.mark.parametrize("failure", [False, True])
def test_pr_plan_uses_base_to_head_and_explicitly_requests_full_on_diff_failure(
    tmp_path: Path,
    lane_registry: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_wide.py").touch()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npython_files = ["test_*.py", "*_test.py"]\n'
    )
    (tmp_path / REGISTRY).parent.mkdir(parents=True)
    (tmp_path / REGISTRY).write_text(json.dumps(lane_registry))
    monkeypatch.setattr(
        lane_planner.subprocess, "check_output", lambda *a, **k: "checkout\n"
    )
    monkeypatch.setattr(lane_planner, "main_successes", lambda *a: {})
    calls = []

    def fake_changes(root: Path, base: str, head: str) -> list[dict[str, Any]]:
        calls.append((base, head))
        if failure:
            raise subprocess.CalledProcessError(128, "git diff")
        return [{"status": "M", "paths": ["tools/wide/a.py"]}]

    monkeypatch.setattr(lane_planner, "git_changes", fake_changes)
    environment = {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_SHA": "checkout",
        "GITHUB_REF": "refs/pull/1/merge",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    payload = {"pull_request": {"base": {"sha": "base"}, "head": {"sha": "head"}}}
    plan = lane_planner.build_plan(tmp_path, environment, payload)
    assert calls == [("base", "head")]
    assert plan["requested_suites"] == ["wide"]
    assert plan["scope"] == ("full" if failure else "pr")
    if failure:
        assert any(reason.startswith("diff_unavailable:") for reason in plan["reasons"])
    else:
        (tmp_path / "tests/test_new.py").touch()
        with pytest.raises(ValueError, match="current source/registry"):
            validate_plan(plan, root=tmp_path)


def test_invalid_or_stale_plan_cannot_silently_defer_tests(
    lane_registry: dict[str, Any],
) -> None:
    plan = make_plan(
        lane_registry,
        ["tests/test_wide.py"],
        source_sha="head",
        event="pull_request",
        base_sha="base",
        head_sha="head",
    )
    plan["requested_suites"] = ["wide"]
    with pytest.raises(ValueError, match="identity"):
        validate_plan(plan)
    from scripts.ci_metrics.lanes import digest

    plan["plan_id"] = digest({k: v for k, v in plan.items() if k != "plan_id"})
    with pytest.raises(ValueError, match="registry/diff"):
        validate_plan(plan)


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
