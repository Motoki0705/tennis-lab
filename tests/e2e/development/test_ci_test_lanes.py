"""Actions must preserve CPU test coverage and propagate pytest failures."""

from __future__ import annotations

import fnmatch
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
    assert set(workflow["jobs"]) == {"plan", "lint", "tests", "system-deps", "python"}
    return cast(dict[str, Any], workflow["jobs"]["tests"])


@pytest.mark.parametrize("event", ["pull_request", "push", "pull_request_target"])
@pytest.mark.parametrize(
    "changes",
    [
        "logs",
        "decision_logs",
        "readmes",
        "existing_event_edit",
        "handwritten_readme",
        "mixed_log_formats",
        "code",
        "mixed",
    ],
)
def test_pr_pushes_run_ci_and_only_pr_runs_cancel_previous_runs(
    event: str, changes: str
) -> None:
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers
    if event == "pull_request_target":
        labeler = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/pull-request-labeler.yml").read_text()
        )
        labeler_events = labeler.get("on", labeler.get(True))
        assert "push" not in labeler_events
        selected = labeler_events[event]
        assert selected["types"] == [
            "opened",
            "synchronize",
            "reopened",
            "ready_for_review",
        ]
    else:
        selected = triggers[event]
    assert "paths" not in selected
    ignored = selected["paths-ignore"]
    assert set(ignored) == {
        f".agents/logs/parents/**/{n}"
        for n in ("instruction.md", "final.md", "meta.json")
    } | {
        ".agents/logs/parents/parent-*/activity-*/event-*.json",
        ".agents/logs/parents/README.md",
        ".agents/logs/parents/parent-*/README.md",
        ".agents/logs/parents/parent-*/activity-*/README.md",
        ".agents/logs/parents/parent-*/child-*/README.md",
        ".agents/logs/parents/parent-*/child-*/d-*/README.md",
        ".agents/logs/parents/parent-*/child-*/d-*/*-*/README.md",
    }
    log = ".agents/logs/parents/parent-1/child-10/d-abc/0001-initial/"
    worker_logs = [log + n for n in ("instruction.md", "final.md", "meta.json")]
    decision_logs = [".agents/logs/parents/parent-1/activity-public/event-public.json"]
    readmes = [
        ".agents/logs/parents/README.md",
        ".agents/logs/parents/parent-1/README.md",
        ".agents/logs/parents/parent-1/activity-public/README.md",
        ".agents/logs/parents/parent-index-legacy-0001/README.md",
        ".agents/logs/parents/parent-1/activity-index-activities-0001/README.md",
        ".agents/logs/parents/parent-1/activity-public-records-0001/README.md",
        ".agents/logs/parents/parent-1/child-10/README.md",
        ".agents/logs/parents/parent-1/child-10/d-abc/README.md",
        ".agents/logs/parents/parent-1/child-10/d-abc/turn-0001/README.md",
        ".agents/logs/parents/parent-1/child-10/d-index-activities-0001/README.md",
        ".agents/logs/parents/parent-1/child-10/d-abc/page-children-0001/README.md",
        log + "README.md",
    ]
    code = ["src/utils/example.py"]
    files = {
        "logs": worker_logs,
        "decision_logs": decision_logs,
        "readmes": readmes,
        # Path filters see neither status nor contents. These invalid publication
        # diffs also skip CI; the ordinary merge gate must reject missing Python.
        "existing_event_edit": decision_logs,
        "handwritten_readme": [readmes[1]],
        "mixed_log_formats": [*worker_logs, *decision_logs, *readmes],
        "code": code,
        "mixed": [*worker_logs, *decision_logs, *readmes, *code],
    }[changes]
    assert any(
        not any(fnmatch.fnmatchcase(p, pattern) for pattern in ignored) for p in files
    ) is (changes in {"code", "mixed"})
    for ordinary in (
        ".agents/ops/worker_logs/schema.py",
        ".agents/ops/decision_logs/publication.py",
        ".agents/ops/decision_logs/README.md",
        ".github/workflows/ci.yml",
        log + "stderr.log",
        ".agents/logs/parents/parent-1/activity-public/private.json",
    ):
        assert not any(fnmatch.fnmatchcase(ordinary, pattern) for pattern in ignored)
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
    assert set(aggregate["needs"]) == {"plan", "lint", "tests", "system-deps"}
    runtime = workflow["jobs"]["system-deps"]
    assert runtime["needs"] == ["plan"]
    assert any(
        "install_system_deps.sh" in step.get("run", "") for step in runtime["steps"]
    )
    assert not any(
        "install_system_deps.sh" in step.get("run", "") for step in job["steps"]
    )
    assert not any(
        "apt-get" in step.get("run", "") or "--with-deps" in step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job["steps"]
    )
    browser = next(
        step
        for step in job["steps"]
        if step["name"] == "Prepare pinned Playwright and Chromium"
    )
    assert "playwright install chromium --only-shell" in browser["run"]
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
    plan_name = "cpu-lane-plan-${{ github.run_id }}"
    assert planning["steps"][-1]["with"]["name"] == plan_name
    assert planning["steps"][-1]["with"]["overwrite"] is True
    for consumer in (job, aggregate):
        download = next(
            step
            for step in consumer["steps"]
            if step["name"] == "Download CPU lane plan"
        )
        assert download["with"]["run-id"] == "${{ github.run_id }}"
        assert download["with"]["github-token"] == "${{ github.token }}"
        if consumer is job:
            assert download["with"]["name"] == plan_name
        else:
            assert (
                download["with"]["artifact-ids"]
                == "${{ steps.artifacts.outputs.plan_artifact_id }}"
            )
    selection = next(
        step for step in aggregate["steps"] if step.get("id") == "artifacts"
    )
    assert selection["env"]["GH_TOKEN"] == "${{ github.token }}"
    selection_command = shlex.split(
        selection["run"].replace("$CI_SHARD_COUNT", default)
    )
    assert selection_command[:3] == ["python3", "-m", "scripts.ci_metrics.artifacts"]
    assert (
        selection_command[selection_command.index("--output") + 1]
        == ".ci-results/artifacts.json"
    )
    assert selection_command[selection_command.index("--count") + 1] == default
    downloads = [
        step
        for step in aggregate["steps"]
        if "download-artifact@" in step.get("uses", "")
    ]
    assert all(
        aggregate["steps"].index(selection) < aggregate["steps"].index(step)
        for step in downloads
    )
    observations = next(
        step for step in downloads if step["name"] == "Download all shard observations"
    )
    assert (
        observations["with"]["artifact-ids"]
        == "${{ steps.artifacts.outputs.shard_artifact_ids }}"
    )
    assert observations["with"]["run-id"] == "${{ github.run_id }}"
    assert observations["with"]["github-token"] == "${{ github.token }}"
    assert observations["with"].get("merge-multiple", False) is False
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
    assert aggregate["steps"][0]["with"]["fetch-depth"] == 0
    assert verification["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert (
        verification_command[verification_command.index("--artifact-manifest") + 1]
        == ".ci-results/artifacts.json"
    )
    assert (
        verification_command[verification_command.index("--timing-profile") + 1]
        == "scripts/ci_metrics/cpu-file-times.json"
    )
    uploaded = aggregate["steps"][-1]["with"]["path"].splitlines()
    assert set(uploaded) == {
        ".ci-results/coverage.json",
        ".ci-results/growth.json",
        ".ci-results/growth-summary.json",
        ".ci-results/growth-summary.md",
        ".ci-results/artifacts.json",
    }
    assert (
        aggregate["steps"][-1]["with"]["name"]
        == "cpu-test-coverage-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert "uv sync --locked --no-group gpu --group cpu" in [
        step.get("run") for step in job["steps"]
    ]


def test_browser_preparation_reaches_pytest_without_ubuntu_packages(
    tmp_path: Path,
) -> None:
    job = _ci_job()
    step = next(
        s for s in job["steps"] if s["name"] == "Prepare pinned Playwright and Chromium"
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    log = tmp_path / "calls.jsonl"
    common = (
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(log)!r}, 'a') as stream: stream.write(json.dumps(sys.argv) + chr(10))\n"
    )
    for name in ("apt-get", "sudo", "npm"):
        path = binary / name
        path.write_text(
            common
            + (
                "sys.exit('Ubuntu unavailable')\n"
                if name != "npm"
                else "assert sys.argv[1:] == ['ci', '--prefix', 'scripts/ci_metrics', '--no-audit', '--no-fund']\n"
            )
        )
        path.chmod(0o755)
    browser = tmp_path / "scripts/ci_metrics/node_modules/.bin/playwright"
    browser.parent.mkdir(parents=True)
    browser.write_text(
        common + "assert sys.argv[1:] == ['install', 'chromium', '--only-shell']\n"
    )
    browser.chmod(0o755)
    (tmp_path / "test_ready.py").write_text("def test_reached_pytest(): pass\n")
    result = subprocess.run(
        [
            "/bin/bash",
            "-euc",
            step["run"]
            + f"\n{shlex.quote(sys.executable)} -m pytest -o addopts='' -q test_ready.py",
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(binary)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [Path(row[0]).name for row in calls] == ["npm", "playwright"]


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
    restored_files = {
        "tests/integration/synthetic_data_generation/test_court_dataset.py",
    }
    assert all((REPO_ROOT / path).is_file() for path in restored_files)
    samples = {
        "tests/unit/test_unit.py": "def test_unit(): pass\n",
        "tests/integration/test_integration.py": "def test_integration(): pass\n",
        "tests/e2e/test_e2e.py": "def test_e2e(): pass\n",
        "tests/newly_added_test.py": f"def test_new(): assert {not failing_test}\n",
        "tests/test_runtime.py": (
            "import pytest\n"
            "@pytest.mark.system_deps\n"
            "def test_real_binary(): pass\n"
            "def test_without_binary(): pass\n"
        ),
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
        **dict.fromkeys(restored_files, "def test_court_contract(): pass\n"),
    }
    for relative, source in samples.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    (tmp_path / REGISTRY).write_text(json.dumps({"schema_version": 1, "suites": []}))
    registry, files = load_registry(tmp_path)
    actions_context = {
        "GITHUB_SHA": "head",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    lane_plan = make_plan(
        registry,
        files,
        source_sha=actions_context["GITHUB_SHA"],
        event="local",
        run_id=actions_context["GITHUB_RUN_ID"],
        attempt=actions_context["GITHUB_RUN_ATTEMPT"],
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
    for index in range(shard_count + 1):
        current_template = (
            template
            if index < shard_count
            else next(
                step["run"]
                for step in yaml.safe_load(
                    (REPO_ROOT / ".github/workflows/ci.yml").read_text()
                )["jobs"]["system-deps"]["steps"]
                if step["name"] == "Test"
            )
        )
        command = shlex.split(
            current_template.replace("${{ matrix.shard }}", str(index))
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
                **actions_context,
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
        measured = tmp_path / (
            f".ci-metrics/shard-{index}/tests"
            if index < shard_count
            else ".ci-metrics/shard-system-deps/tests"
        )
        assert (measured / "pytest.log").read_text() in output
        summary = json.loads((measured / "summary.json").read_text())
        assert summary["issues"] == []
        assert summary["successful_test_baseline"] == (completed.returncode == 0)
        assert summary["full_cpu_baseline"] is False
        selected = json.loads((measured / "inventory.json").read_text())
        if index == shard_count:
            assert selected == ["tests/test_runtime.py::test_real_binary"]
            assert summary["counts"] == {"passed": 1}
        else:
            assert "tests/test_runtime.py::test_real_binary" not in selected
        failures += summary["counts"].get("failed", 0)
        inventory.extend(json.loads((measured / "inventory.json").read_text()))
        results.extend(json.loads((measured / "results.json").read_text()))
        summaries.append(summary)
    assert len(inventory) == len(set(inventory)) == (14 if failing_test else 13)
    assert any(
        "test_court_dataset.py::test_court_contract" in node for node in inventory
    )
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
        *restored_files,
        "tests/unit/test_unit.py",
        "tests/integration/test_integration.py",
        "tests/e2e/test_e2e.py",
        "tests/newly_added_test.py",
        "tests/test_setup_outcomes.py",
        "tests/test_runtime.py",
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
        "non_list_paths",
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
    elif fault == "non_list_paths":
        suite["paths"] = None
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
    assert (
        by_name["Upload test measurements"]["with"]["name"]
        == "cpu-test-measurements-${{ github.run_id }}-shard-${{ matrix.shard }}"
    )
    assert by_name["Upload test measurements"]["with"]["overwrite"] is True
    assert (
        '--attempt "$GITHUB_RUN_ATTEMPT"' in by_name["Snapshot attempt timings"]["run"]
    )
    assert [step["name"] for step in steps].index("Record runner environment") < [
        step["name"] for step in steps
    ].index("Install locked CPU environment")


@pytest.mark.parametrize(
    "scenario, expected, reason",
    [
        ("download", 0, None),
        ("cached", 0, None),
        ("corrupt_download", 1, "SHA-256 mismatch"),
        ("corrupt_cache", 1, "SHA-256 mismatch"),
        ("download_failure", 42, "download unavailable"),
        ("timeout", 28, "download timed out"),
        ("bad_archive", 1, "preparation failed"),
        ("version_failure", 44, "broken executable"),
        ("missing_codec", 1, "libx264"),
        ("signature_failure", 43, "invalid signature"),
        ("wrong_signer", 1, "pinned signer"),
        ("signed_pin_mismatch", 1, "differs from signed SHA256SUMS"),
    ],
)
def test_system_dependencies_are_verified_and_failures_are_not_hidden(
    tmp_path: Path, scenario: str, expected: int, reason: str | None
) -> None:
    import hashlib
    import io
    import tarfile
    import zipfile

    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "python3").symlink_to(sys.executable)
    log = tmp_path / "calls.jsonl"
    manifest = json.loads(
        (REPO_ROOT / "scripts/ci_metrics/system-deps.json").read_text()
    )
    archives = tmp_path / "archives"
    archives.mkdir()
    for row in manifest["binaries"]:
        source = (
            f"#!{sys.executable}\nimport sys\n"
            + (
                "print('broken executable', file=sys.stderr); sys.exit(44)\n"
                if scenario == "version_failure"
                else ""
            )
            + (
                f"print({('' if scenario == 'missing_codec' else 'libx264')!r} if '-encoders' in sys.argv else 'ffmpeg version {row['version']} fixture')\n"
                if row["name"] == "ffmpeg"
                else f"print('rclone {row['version']}')\n"
            )
        ).encode()
        archive = archives / row["url"].rsplit("/", 1)[-1]
        if row["format"] == "tar.xz":
            with tarfile.open(archive, "w:xz") as tar:
                info = tarfile.TarInfo(row["member"])
                info.size = len(source)
                tar.addfile(info, io.BytesIO(source))
        else:
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr(row["member"], source)
        if scenario == "bad_archive" and row["name"] == "ffmpeg":
            archive.write_bytes(b"not an archive")
        row["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest["rclone_sums"] = "sums"
    manifest["rclone_key"] = "key"
    (tmp_path / "key").write_text("fixture key verified by fake gpg")
    rclone = manifest["binaries"][1]
    (tmp_path / "sums").write_text(
        f"{rclone['sha256']}  {rclone['url'].rsplit('/', 1)[-1]}\n"
    )
    if scenario == "signed_pin_mismatch":
        rclone["sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    common = (
        "import json, os, sys, shutil\nfrom pathlib import Path\n"
        f"scenario = {scenario!r}\n"
        f"with open({str(log)!r}, 'a') as stream: "
        "stream.write(json.dumps(sys.argv) + chr(10))\n"
    )

    def executable(name: str, source: str) -> None:
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + common + source)
        path.chmod(0o755)

    executable("sudo", "sys.exit('Ubuntu unavailable: sudo must never run')\n")
    executable("apt-get", "sys.exit('Ubuntu unavailable: apt must never run')\n")
    executable("ffmpeg", "sys.exit('unverified host binary must never run')\n")
    executable("rclone", "sys.exit('unverified host binary must never run')\n")
    executable(
        "curl",
        "assert sys.argv[sys.argv.index('--max-time') + 1] == '180'\n"
        "assert sys.argv[sys.argv.index('--retry') + 1] == '0'\n"
        "assert sys.argv[sys.argv.index('--proto-redir') + 1] == '=https'\n"
        "if scenario == 'download_failure': print('download unavailable', file=sys.stderr); sys.exit(42)\n"
        "if scenario == 'timeout': print('download timed out', file=sys.stderr); sys.exit(28)\n"
        "destination = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        f"shutil.copyfile(Path({str(archives)!r}) / sys.argv[-1].rsplit('/', 1)[-1], destination)\n"
        "if scenario == 'corrupt_download': destination.write_bytes(b'corrupt')\n",
    )
    executable(
        "gpg",
        "assert '--no-auto-key-retrieve' in sys.argv\n"
        "if '--decrypt' in sys.argv:\n"
        "    if scenario == 'signature_failure': print('invalid signature', file=sys.stderr); sys.exit(43)\n"
        f"    print('[GNUPG:] VALIDSIG {('0' * 40 if scenario == 'wrong_signer' else manifest['rclone_signer'])} 2026 fixture')\n"
        "    shutil.copyfile(sys.argv[-1], sys.argv[sys.argv.index('--output') + 1])\n",
    )
    cache = tmp_path / "cache"
    if scenario in {"cached", "corrupt_cache"}:
        shutil.copytree(archives, cache)
        if scenario == "corrupt_cache":
            (cache / manifest["binaries"][0]["url"].rsplit("/", 1)[-1]).write_bytes(
                b"corrupt"
            )
    output = tmp_path / "installed"
    github_path = tmp_path / "github-path"
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        s
        for s in workflow["jobs"]["system-deps"]["steps"]
        if s["name"] == "Install pinned system dependencies"
    )
    assert step["timeout-minutes"] == 8
    completed = subprocess.run(
        [
            "/bin/bash",
            *shlex.split(step["run"])[1:],
            "--manifest",
            str(manifest_path),
            "--cache",
            str(cache),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": str(binary), "GITHUB_PATH": str(github_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    if reason:
        assert reason in completed.stdout + completed.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any(
        Path(row[0]).name in {"apt-get", "sudo", "ffmpeg", "rclone"} for row in calls
    )
    if scenario in {
        "cached",
        "corrupt_cache",
        "signature_failure",
        "wrong_signer",
        "signed_pin_mismatch",
    }:
        assert not any(Path(row[0]).name == "curl" for row in calls)
    if expected == 0:
        assert github_path.read_text().strip() == str(output / "bin")
        assert (output / "bin/ffmpeg").is_file() and (output / "bin/rclone").is_file()
        assert "Verified rclone SHA256SUMS signature" in completed.stdout
    else:
        assert not github_path.exists()
        assert not (output / "bin").exists()


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
