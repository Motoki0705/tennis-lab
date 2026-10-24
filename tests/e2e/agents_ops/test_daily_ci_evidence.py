"""Real git, pytest and xdist evidence; no Actions dispatch or real-time override CLI."""

from __future__ import annotations

import copy
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
ev: Any = importlib.import_module("daily_ci.evidence")
wf: Any = importlib.import_module("daily_ci.workflow")
model: Any = importlib.import_module("daily_ci.model")


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


@pytest.fixture
def job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "fixture")
    git(root, "config", "user.email", "fixture@example.com")
    (root / "old.txt").write_text("baseline deletion must remain visible")
    git(root, "add", ".")
    git(root, "commit", "-qm", "last successful baseline")
    baseline = git(root, "rev-parse", "HEAD")
    git(root, "rm", "old.txt")
    git(root, "commit", "-qm", "failed previous day")
    git(root, "commit", "--allow-empty", "-qm", "new day")
    target = git(root, "rev-parse", "HEAD")
    git(root, "update-ref", "refs/remotes/origin/main", target)
    identity = {
        "repository": "o/r",
        "workflow": "ci.yml",
        "job": "lint-and-test",
        "shard": "all",
        "cycle_id": "daily-ci-2026-10-25",
        "generation": 1,
        "request_id": model.request_id("o/r", "daily-ci-2026-10-25", 1, "ci.yml"),
        "target_sha": target,
        "checkout_sha": target,
        "baseline_sha": baseline,
        "deadline_at": "2026-10-25T09:00:00+09:00",
        "run_id": 123,
        "attempt": 1,
    }
    env = {
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": target,
        "GITHUB_WORKFLOW_SHA": target,
        "GITHUB_WORKFLOW_REF": "o/r/.github/workflows/ci.yml@refs/heads/main",
        "GITHUB_REPOSITORY": "o/r",
        "GITHUB_JOB": "lint-and-test",
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "PYTHONPATH": str(OPS),
        **{"DAILY_CI_" + k.upper(): str(v) for k, v in identity.items()},
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(
        wf, "utcnow", lambda: datetime.fromisoformat("2026-10-25T06:05:00+09:00")
    )
    path = root / "evidence/manifest.json"
    wf.initialize(path, root, dict(os.environ))
    return root, path, identity


def fill_checks(root: Path, path: Path, *, lint_code: int = 0) -> None:
    for name in ("environment", "cpu-backend", "system-dependencies", "lint"):
        code = lint_code if name == "lint" else 0
        assert (
            wf.run_check(
                path,
                root,
                name,
                [sys.executable, "-c", f"raise SystemExit({code})"],
                root,
            )
            == code
        )


@pytest.mark.parametrize("workers", [0, 2])
def test_real_reports_preserve_parameter_spaces_skips_failures_and_teardown(
    job: tuple[Path, Path, dict[str, Any]], workers: int
) -> None:
    root, path, identity = job
    (root / "test_sample.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('x', [1,2], ids=['a b', 'a::b [c]'])\n"
        "def test_parameter(x): pass\n"
        "def test_failure(): assert False\n"
        "def test_skip(): pytest.skip('not executed')\n"
        "@pytest.mark.xfail(reason='known')\n"
        "def test_xpass(): pass\n"
        "@pytest.fixture\n"
        "def broken_teardown():\n"
        "    yield\n"
        "    raise ValueError('teardown')\n"
        "def test_teardown(broken_teardown): pass\n"
    )
    fill_checks(root, path)
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(root),
        "-o",
        "addopts=",
        "-q",
        "-n",
        str(workers),
        "--dist=worksteal",
        "-p",
        "daily_ci.pytest_evidence",
        "--junitxml=" + str(path.parent / "pytest.junit.xml"),
        "test_sample.py",
    ]
    assert wf.run_check(path, root, "test", argv, root) == 1
    wf.finalize(path, root)
    evidence = ev.validate_bundle(path.parent, identity, remote_conclusion="failure")
    assert evidence["result"] == "failure"
    assert ev.passed_nodeids(evidence) == {
        "test_sample.py::test_parameter[a b]",
        "test_sample.py::test_parameter[a::b [c]]",
    }
    assert "test_sample.py::test_renamed" not in ev.passed_nodeids(evidence)
    assert evidence["tests"]["test_sample.py::test_teardown"] == "failed"
    assert evidence["tests"]["test_sample.py::test_skip"] == "skipped"
    assert evidence["tests"]["test_sample.py::test_xpass"] == "skipped"
    # Attempts, generations and SHA identity are never merged across artifacts.
    for key, value in (
        ("attempt", 2),
        ("target_sha", "a" * 40),
        ("request_id", "f" * 32),
    ):
        with pytest.raises(ValueError):
            ev.validate_bundle(
                path.parent, {**identity, key: value}, remote_conclusion="success"
            )
    for conclusion in ("cancelled", "skipped", "neutral", "timed_out"):
        with pytest.raises(ValueError, match="remote job"):
            ev.validate_bundle(path.parent, identity, remote_conclusion=conclusion)
    junit = path.parent / "pytest.junit.xml"
    junit.write_text(junit.read_text() + "modified")
    with pytest.raises(ValueError, match="modified"):
        ev.validate_bundle(path.parent, identity, remote_conclusion="failure")
    junit.unlink()
    with pytest.raises(ValueError, match="missing"):
        ev.validate_bundle(path.parent, identity, remote_conclusion="failure")


def test_lint_failure_and_cancelled_execution_do_not_invent_test_success(
    job: tuple[Path, Path, dict[str, Any]],
) -> None:
    root, path, identity = job
    fill_checks(root, path, lint_code=1)
    wf.finalize(path, root)
    evidence = ev.validate_bundle(path.parent, identity, remote_conclusion="failure")
    assert evidence["checks"]["test"]["outcome"] == "not_run"
    assert not evidence["tests"]
    manifest = ev.read_json(path)
    manifest["checks"]["test"]["outcome"] = "incomplete"
    manifest["finished_at"] = None  # killed before finalizer/upload
    wf.save(path, manifest)
    with pytest.raises(ValueError, match="unfinished"):
        ev.validate_bundle(path.parent, identity, remote_conclusion="failure")


@pytest.mark.parametrize(
    "time",
    [
        "2026-10-25T05:59:59+09:00",
        "2026-10-25T09:00:00+09:00",
        "2026-10-26T06:01:00+09:00",
    ],
)
def test_guard_rejects_early_expired_and_stale_cycle(
    job: tuple[Path, Path, dict[str, Any]], time: str
) -> None:
    root, _, identity = job
    with pytest.raises(ValueError, match="window"):
        wf.guard(identity, root, dict(os.environ), datetime.fromisoformat(time))


def test_guard_binds_real_head_main_baseline_and_ref(
    job: tuple[Path, Path, dict[str, Any]],
) -> None:
    root, _, identity = job
    now = wf.utcnow()
    assert (
        git(
            root,
            "diff",
            "--name-status",
            identity["baseline_sha"],
            identity["target_sha"],
        )
        == "D\told.txt"
    )
    # New day's target keeps the failed day's deletion, not just HEAD~1..HEAD.
    wf.guard(identity, root, dict(os.environ), now)
    for key, value in (
        ("generation", 0),
        ("deadline_at", "2026-10-25T10:00:00+09:00"),
        ("target_sha", "$(touch /tmp/injected)"),
    ):
        with pytest.raises(ValueError):
            wf.guard({**identity, key: value}, root, dict(os.environ), now)
    for key, value in (
        ("GITHUB_SHA", "a" * 40),
        ("GITHUB_REF", "refs/heads/evil"),
        ("GITHUB_EVENT_NAME", "push"),
    ):
        with pytest.raises(ValueError, match="mismatch"):
            wf.guard(identity, root, {**os.environ, key: value}, now)
    git(root, "update-ref", "refs/remotes/origin/main", identity["baseline_sha"])
    with pytest.raises(ValueError, match="ancestor"):
        wf.guard(identity, root, dict(os.environ), now)
    git(root, "checkout", "--detach", identity["baseline_sha"])
    with pytest.raises(ValueError, match="mismatch"):
        wf.guard(identity, root, dict(os.environ), now)


def test_finish_after_deadline_is_rejected_but_artifacts_survive(
    job: tuple[Path, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path, identity = job
    fill_checks(root, path)
    monkeypatch.setattr(
        wf, "utcnow", lambda: datetime.fromisoformat(identity["deadline_at"])
    )
    with pytest.raises(ValueError, match="window"):
        wf.finalize(path, root)
    assert (path.parent / "checks.junit.xml").exists()
    with pytest.raises(ValueError, match="unvalidated"):
        ev.validate_bundle(path.parent, identity, remote_conclusion="success")


def test_workflow_contract_preserves_jobs_and_dispatch_only() -> None:
    for workflow, jobs in (
        ("ci.yml", {"lint-and-test"}),
        ("knowledge.yml", {"knowledge", "knowledge-webui"}),
    ):
        raw = (ROOT / ".github/workflows" / workflow).read_text()
        doc = yaml.safe_load(raw)
        assert set(doc[True]) == {"workflow_dispatch"}
        assert set(doc[True]["workflow_dispatch"]["inputs"]) == {
            "cycle_id",
            "generation",
            "request_id",
            "target_sha",
            "baseline_sha",
            "deadline_at",
        }
        assert doc["run-name"] == "daily-ci:${{ inputs.request_id }}"
        assert set(doc["jobs"]) == jobs
        assert doc["concurrency"]["cancel-in-progress"] is False
        for name, job in doc["jobs"].items():
            steps = job["steps"]
            assert steps[0]["with"]["ref"] == "${{ inputs.target_sha }}"
            assert steps[0]["with"]["fetch-depth"] == 0
            assert steps[1]["run"] == "python3 -m daily_ci.workflow init"
            assert steps[-2]["if"] == steps[-1]["if"] == "always()"
            assert steps[-1]["with"]["if-no-files-found"] == "error"
            commands = [s["run"] for s in steps if "run" in s]
            assert all("${{" not in command for command in commands)
            assert all(
                any(f" {check} -- " in command for command in commands)
                for check in ev.PLANS[workflow, name, "all"]
            )
        assert "event.before" not in raw and "pull_request" not in raw
    knowledge = (ROOT / ".github/workflows/knowledge.yml").read_text()
    assert '--base-ref "$DAILY_CI_BASELINE_SHA"' in knowledge
    for name in ("local-gpu-tests.yml", "local-gpu-training.yml"):
        assert set(
            yaml.safe_load((ROOT / ".github/workflows" / name).read_text())[True]
        ) == {"workflow_dispatch"}
    assert (
        "pull_request_target"
        in (ROOT / ".github/workflows/pull-request-labeler.yml").read_text()
    )


def test_checked_in_bundle_fixture_is_consumable() -> None:
    path = Path(__file__).parent / "fixtures/daily_ci/knowledge-webui"
    identity = ev.read_json(path / "manifest.json")["identity"]
    assert (
        ev.validate_bundle(path, identity, remote_conclusion="success")["result"]
        == "success"
    )
    bad = copy.deepcopy(identity)
    bad["job"] = "knowledge"
    with pytest.raises(ValueError):
        ev.validate_bundle(path, bad, remote_conclusion="success")
    common = {k: v for k, v in identity.items() if k not in {"job", "shard"}}
    with pytest.raises(ValueError, match="missing"):
        ev.validate_workflow(
            {("knowledge-webui", "all"): path},
            common,
            {("knowledge-webui", "all"): "success"},
        )


def test_python_bundle_fixture_preserves_exact_recovery_ids() -> None:
    path = Path(__file__).parent / "fixtures/daily_ci/python"
    identity = ev.read_json(path / "manifest.json")["identity"]
    common = {k: v for k, v in identity.items() if k not in {"job", "shard"}}
    result = ev.validate_workflow(
        {("lint-and-test", "all"): path}, common, {("lint-and-test", "all"): "success"}
    )
    assert result["result"] == "success"
    assert ev.passed_nodeids(result["jobs"]["lint-and-test", "all"]) == {
        "test_cases.py::test_value[a b :: [c]]"
    }


def test_commands_keep_metacharacters_as_literal_arguments(
    job: tuple[Path, Path, dict[str, Any]],
) -> None:
    root, path, _ = job
    value = '$(touch INJECTED); `touch INJECTED` " spaced value'
    argv = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; Path('literal').write_text(sys.argv[1])",
        value,
    ]
    assert wf.run_check(path, root, "environment", argv, root) == 0
    assert (root / "literal").read_text() == value
    assert not (root / "INJECTED").exists()
    assert ev.read_json(path)["checks"]["environment"]["argv"] == argv


def test_deadline_rechecked_after_start_journal_before_process(
    job: tuple[Path, Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, path, identity = job
    original = wf.save

    def save(path: Path, value: dict[str, Any]) -> None:
        original(path, value)
        monkeypatch.setattr(
            wf, "utcnow", lambda: datetime.fromisoformat(identity["deadline_at"])
        )

    monkeypatch.setattr(wf, "save", save)
    with pytest.raises(ValueError, match="window"):
        wf.run_check(
            path,
            root,
            "environment",
            [sys.executable, "-c", "from pathlib import Path; Path('late').touch()"],
            root,
        )
    assert not (root / "late").exists()
    assert ev.read_json(path)["checks"]["environment"]["outcome"] == "incomplete"
