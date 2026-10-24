"""Real CLI + SQLite + fake gh: collection, retries, classification and read boundary."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
FAKE = Path(__file__).with_name("fake_ci_gh.py")
TEST_LOG = "2026-10-24T01:00:00Z FAILED tests/test_scene.py::test_tracks - AssertionError: tracks differ\n"


def sha(number: int) -> str:
    return f"{number:040x}"


class CIEnv:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.state: dict[str, Any] = {
            "head": sha(1),
            "commits": {},
            "runs": {},
            "attempts": {},
            "jobs": {},
            "logs": {},
            "errors": {},
            "overrides": {},
        }
        self.remote = tmp / "remote.json"
        self.calls = tmp / "calls.jsonl"
        self.local = tmp / "xdg/tennis-lab-agents/ci-watch"
        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(
            f"#!{sys.executable}\nexec(compile(open({str(FAKE)!r}).read(), {str(FAKE)!r}, 'exec'))\n"
        )
        gh.chmod(0o700)
        self.env = {
            **os.environ,
            "PYTHONPATH": str(ROOT / ".agents/ops"),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "XDG_STATE_HOME": str(tmp / "xdg"),
            "FAKE_CI_STATE": str(self.remote),
            "FAKE_CI_CALLS": str(self.calls),
        }
        self.commit(1, None)

    def commit(self, number: int, parent: int | None) -> None:
        self.state["head"] = sha(number)
        self.state["commits"][sha(number)] = {
            "sha": sha(number),
            "parents": [{"sha": sha(parent)}] if parent else [],
            "html_url": f"https://github.com/o/r/commit/{sha(number)}",
            "commit": {"committer": {"date": "2026-10-24T01:00:00Z"}},
        }

    def run_fixture(
        self,
        commit: int,
        conclusion: str | None = "success",
        *,
        attempt: int = 1,
        workflow: str = "ci.yml",
        run_id: int | None = None,
        log: str = TEST_LOG,
        step: str = "Test",
    ) -> dict[str, Any]:
        identifier = run_id or commit * 100
        row = {
            "id": identifier,
            "run_attempt": attempt,
            "head_sha": sha(commit),
            "head_branch": "main",
            "event": "push",
            "path": f".github/workflows/{workflow}",
            "status": "completed" if conclusion else "in_progress",
            "conclusion": conclusion,
            "html_url": f"https://github.com/o/r/actions/runs/{identifier}",
        }
        runs = self.state["runs"].setdefault(sha(commit), {}).setdefault(workflow, [])
        runs[:] = [r for r in runs if r["id"] != identifier]
        runs.append(row.copy())
        key = f"{identifier}/{attempt}"
        self.state["attempts"][key] = row
        job = self.job(
            identifier,
            commit,
            attempt,
            identifier * 1000 + attempt,
            "Python",
            conclusion,
            step,
        )
        self.state["jobs"][key] = [job]
        self.state["logs"][str(job["id"])] = log
        return row

    @staticmethod
    def job(
        run: int,
        commit: int,
        attempt: int,
        identifier: int,
        name: str,
        conclusion: str | None,
        step: str,
    ) -> dict[str, Any]:
        status = "completed" if conclusion else "in_progress"
        return {
            "id": identifier,
            "run_id": run,
            "run_attempt": attempt,
            "head_sha": sha(commit),
            "name": name,
            "status": status,
            "conclusion": conclusion,
            "html_url": f"https://github.com/o/r/actions/runs/{run}/job/{identifier}",
            "steps": [
                {"number": 1, "name": step, "status": status, "conclusion": conclusion}
            ],
        }

    def cli(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        self.remote.write_text(json.dumps(self.state))
        return subprocess.run(
            [sys.executable, "-m", "ci_watch", *args],
            cwd=ROOT,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def collect(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.cli("collect", "--repo", "o/r", *args, **env)

    def start(self, *args: str) -> subprocess.CompletedProcess[str]:
        return self.collect("--start-sha", sha(1), *args)

    def show(self, number: int) -> dict[str, Any]:
        result = self.cli("show", "--sha", sha(number))
        assert result.returncode == 0, result.stderr
        value: dict[str, Any] = json.loads(result.stdout)["observation"]
        return value

    def records(self, *args: str) -> dict[str, Any]:
        result = self.cli("records", *args)
        assert result.returncode == 0, result.stderr
        value: dict[str, Any] = json.loads(result.stdout)
        return value

    def endpoints(self) -> list[str]:
        return [json.loads(line)[-1] for line in self.calls.read_text().splitlines()]


@pytest.fixture
def ci(tmp_path: Path) -> CIEnv:
    return CIEnv(tmp_path)


def test_sha_cursor_pending_errors_resume_and_idempotence(ci: CIEnv) -> None:
    first = ci.start()
    assert first.returncode == 0, first.stderr
    assert json.loads(first.stdout)["complete"] is False
    assert ci.show(1)["workflows"][0]["status"] == "no_run"
    ci.commit(2, 1)
    ci.commit(3, 2)
    ci.run_fixture(1)
    ci.run_fixture(2, None)
    ci.run_fixture(3, "failure")
    result = ci.collect()
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["discovered"] == 2 and summary["cursor_sha"] == sha(3)
    assert summary["pending"] == [sha(2)]
    assert ci.show(2)["workflows"][0]["status"] == "incomplete"
    assert ci.show(3)["result"] == "non_success"
    records = ci.records()["records"]
    again = ci.collect()
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)["appended"] == 0
    assert ci.records()["records"] == records
    ci.run_fixture(2)
    ci.commit(4, 3)
    ci.run_fixture(4)
    resumed = ci.collect()
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["discovered"] == 1
    assert json.loads(resumed.stdout)["complete"] is True
    assert ci.show(2)["result"] == "success"
    with sqlite3.connect(ci.local / "observations.sqlite3") as db:
        assert db.execute("SELECT sha FROM commits ORDER BY ordinal").fetchall() == [
            (sha(n),) for n in range(1, 5)
        ]


def test_api_failure_is_recorded_and_retry_does_not_skip_sha(ci: CIEnv) -> None:
    ci.commit(2, 1)
    ci.commit(3, 2)
    for n in range(1, 4):
        ci.run_fixture(n)
    endpoint = (
        "repos/o/r/actions/workflows/ci.yml/runs?branch=main&head_sha="
        + sha(2)
        + "&per_page=100&page=1"
    )
    ci.state["errors"][endpoint] = "HTTP 403: API rate limit exceeded"
    result = ci.start()
    assert result.returncode == 1
    assert ci.show(1)["result"] == ci.show(3)["result"] == "success"
    failed = ci.show(2)
    assert failed["status"] == "fetch_error" and failed["result"] is None
    assert "403" in failed["errors"][0]["reason"]
    boundary = ci.records()["next_after"]
    ci.state["errors"].clear()
    result = ci.collect()
    assert result.returncode == 0, result.stderr
    assert ci.show(2)["result"] == "success"
    assert ci.records("--after", str(boundary))["records"][0]["sha"] == sha(2)
    assert ci.records()["records"][-1]["kind"] == "collection"
    assert ci.records()["records"][-1]["observation"]["status"] == "complete"


def test_discovery_failure_history_rewrite_and_first_parent(ci: CIEnv) -> None:
    ci.commit(2, 1)
    ci.commit(3, 2)
    ci.state["commits"][sha(3)]["parents"].append({"sha": sha(99)})
    ci.state["errors"][f"repos/o/r/commits/{sha(2)}"] = "HTTP 500"
    assert ci.start().returncode == 1
    assert ci.records()["records"][0]["kind"] == "collection"
    ci.state["errors"].clear()
    assert ci.collect().returncode == 0
    assert f"repos/o/r/commits/{sha(99)}" not in ci.endpoints()
    ci.commit(4, 1)
    rewritten = ci.collect()
    assert rewritten.returncode == 1
    assert "history rewrite" in rewritten.stdout
    assert ci.show(3)["status"] == "pending"
    assert ci.cli("show", "--sha", sha(4)).returncode == 1


def test_interruption_after_cursor_commit_can_resume(ci: CIEnv) -> None:
    ci.run_fixture(1)
    endpoint = f"repos/o/r/actions/workflows/ci.yml/runs?branch=main&head_sha={sha(1)}&per_page=100&page=1"
    interrupted = ci.collect("--start-sha", sha(1), FAKE_CI_INTERRUPT=endpoint)
    assert interrupted.returncode != 0
    assert ci.show(1)["status"] == "not_collected"
    assert ci.collect().returncode == 0
    assert ci.show(1)["result"] == "success"


def test_run_and_job_page_boundaries(ci: CIEnv) -> None:
    valid = ci.run_fixture(1)
    ignored = [{**valid, "id": n + 1000, "event": "pull_request"} for n in range(100)]
    ci.state["runs"][sha(1)]["ci.yml"] = ignored + [valid]
    ci.state["jobs"]["100/1"] = [
        ci.job(100, 1, 1, n + 1, f"Python ({n})", "success", "Test") for n in range(101)
    ]
    result = ci.start()
    assert result.returncode == 0, result.stderr
    assert ci.show(1)["result"] == "success"
    assert len(ci.show(1)["workflows"][0]["runs"][0]["jobs"]) == 101
    endpoints = ci.endpoints()
    assert any("/runs?" in e and e.endswith("page=2") for e in endpoints)
    assert any("/jobs?" in e and e.endswith("page=2") for e in endpoints)
    assert not any("/attempts/" in e and "/100/" not in e for e in endpoints)


@pytest.mark.parametrize(
    "shape",
    [
        "short_page",
        "duplicate",
        "foreign_sha",
        "bad_attempt",
        "foreign_job",
        "search_cap",
        "inconsistent_success",
    ],
)
def test_malformed_or_truncated_api_is_not_success(ci: CIEnv, shape: str) -> None:
    ci.run_fixture(1)
    endpoint = f"repos/o/r/actions/workflows/ci.yml/runs?branch=main&head_sha={sha(1)}&per_page=100&page=1"
    row = ci.state["runs"][sha(1)]["ci.yml"][0]
    if shape == "short_page":
        ci.state["overrides"][endpoint] = {"total_count": 2, "workflow_runs": [row]}
    elif shape == "duplicate":
        ci.state["overrides"][endpoint] = {
            "total_count": 2,
            "workflow_runs": [row, row],
        }
    elif shape == "foreign_sha":
        row["head_sha"] = sha(99)
    elif shape == "bad_attempt":
        ci.state["attempts"]["100/1"]["run_attempt"] = 2
    elif shape == "foreign_job":
        ci.state["jobs"]["100/1"][0]["head_sha"] = sha(99)
    elif shape == "search_cap":
        ci.state["overrides"][endpoint] = {"total_count": 1000, "workflow_runs": []}
    else:
        ci.state["jobs"]["100/1"][0]["conclusion"] = "failure"
    result = ci.start()
    assert result.returncode == 1
    assert ci.show(1)["status"] == "fetch_error"
    assert ci.show(1)["result"] is None


def test_persistent_specific_failure_and_no_bridge_across_missing_run(
    ci: CIEnv,
) -> None:
    ci.commit(2, 1)
    ci.run_fixture(1, "failure")
    ci.run_fixture(2, "failure")
    assert ci.start().returncode == 0
    for n in (1, 2):
        failure = ci.show(n)["failures"][0]
        assert failure["test_id"] == "tests/test_scene.py::test_tracks"
        classification = failure["classification"]
        assert classification["category"] == "code_regression"
        assert classification["confidence"] == "medium"
        assert classification["assessment"] == "hypothesis"
        assert classification["reasons"][0]["evidence"][0]["sha"] == sha(3 - n)
    ci.state["runs"][sha(2)]["ci.yml"] = []
    ci.commit(3, 2)
    ci.run_fixture(3, "failure")
    assert ci.collect().returncode == 0
    assert ci.show(1)["failures"][0]["classification"]["category"] == "unclassified"
    assert ci.show(3)["failures"][0]["classification"]["category"] == "unclassified"


def test_same_sha_retry_success_is_flaky_and_keeps_old_attempt(ci: CIEnv) -> None:
    ci.run_fixture(1, "failure")
    assert ci.start().returncode == 0
    original = ci.show(1)["failures"][0]["signature"]
    ci.run_fixture(1, "success", attempt=2)
    assert ci.collect().returncode == 0
    observed = ci.show(1)
    assert observed["result"] == "success"
    assert [r["attempt"] for r in observed["workflows"][0]["runs"]] == [1, 2]
    failure = observed["failures"][0]
    assert failure["signature"] == original
    assert failure["classification"]["category"] == "flaky"
    assert failure["classification"]["confidence"] == "medium"


def test_flaky_neighbor_is_counterevidence_to_regression(ci: CIEnv) -> None:
    ci.commit(2, 1)
    ci.run_fixture(1, "failure")
    ci.run_fixture(1, "success", attempt=2)
    ci.run_fixture(2, "failure")
    assert ci.start().returncode == 0
    assert ci.show(1)["failures"][0]["classification"]["category"] == "flaky"
    failure = ci.show(2)["failures"][0]
    assert failure["classification"]["category"] == "unclassified"
    assert failure["classification"]["reasons"][0]["evidence"]


def test_pytest_parameter_with_spaces_and_multiple_failures(ci: CIEnv) -> None:
    log = (
        TEST_LOG
        + "FAILED tests/test_scene.py::test_tracks[empty input] - ValueError: empty\n"
        + TEST_LOG
    )
    ci.run_fixture(1, "failure", log=log)
    assert ci.start().returncode == 0
    failures = ci.show(1)["failures"]
    assert len(failures) == 2
    assert failures[1]["test_id"] == "tests/test_scene.py::test_tracks[empty input]"


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_skipped_job_is_not_retry_success(ci: CIEnv, conclusion: str) -> None:
    ci.run_fixture(1, "failure")
    ci.run_fixture(1, conclusion, attempt=2)
    assert ci.start().returncode == 0
    assert ci.show(1)["failures"][0]["classification"]["category"] == "unclassified"
    assert ci.show(1)["result"] == "non_success"


@pytest.mark.parametrize(
    ("step", "log", "category"),
    [
        (
            "Install locked CPU environment",
            "error: Failed to download dependency: connection reset\n",
            "runner_infrastructure",
        ),
        ("Test", "RuntimeError: Could not resolve host\n", "unclassified"),
        (
            "Test",
            "The runner has lost communication with the server\n",
            "runner_infrastructure",
        ),
        ("Test", "unexpected opaque crash\n", "unclassified"),
    ],
)
def test_classification_representatives(
    ci: CIEnv, step: str, log: str, category: str
) -> None:
    ci.run_fixture(1, "failure", log=log, step=step)
    assert ci.start().returncode == 0
    failure = ci.show(1)["failures"][0]
    assert failure["classification"]["category"] == category
    assert failure["evidence"]


def test_log_failure_is_explicit_and_cached_log_survives_expiry(ci: CIEnv) -> None:
    ci.run_fixture(1, "failure")
    endpoint = "repos/o/r/actions/jobs/100001/logs"
    ci.state["errors"][endpoint] = "HTTP 410: logs expired"
    assert ci.start().returncode == 1
    assert ci.show(1)["status"] == "fetch_error"
    assert ci.show(1)["failures"][0]["classification"]["category"] == "unclassified"
    ci.state["errors"].clear()
    assert ci.collect().returncode == 0
    ci.state["errors"]["repos/o/r/actions/workflows/ci.yml/runs"] = "HTTP 503"
    assert ci.collect().returncode == 1
    ci.state["errors"].clear()
    ci.state["errors"][endpoint] = "HTTP 410: logs expired"
    assert ci.collect().returncode == 0
    assert ci.show(1)["failures"][0]["test_id"] is not None
    assert ci.endpoints().count(endpoint) == 2


def test_multiple_workflows_and_run_without_jobs(ci: CIEnv) -> None:
    ci.run_fixture(1)
    result = ci.start("--workflow", "ci.yml", "--workflow", "cuda.yml")
    assert result.returncode == 0
    assert ci.show(1)["status"] == "pending" and ci.show(1)["result"] is None
    ci.run_fixture(1, "startup_failure", workflow="cuda.yml", run_id=200)
    ci.state["jobs"]["200/1"] = []
    assert ci.collect("--workflow", "ci.yml", "--workflow", "cuda.yml").returncode == 0
    assert ci.show(1)["result"] == "non_success"
    assert ci.show(1)["failures"][0]["job"] == "<workflow>"


def test_failed_run_with_only_successful_jobs_keeps_workflow_failure(ci: CIEnv) -> None:
    ci.run_fixture(1, "failure")
    job = ci.state["jobs"]["100/1"][0]
    job["conclusion"] = job["steps"][0]["conclusion"] = "success"
    assert ci.start().returncode == 0
    assert ci.show(1)["result"] == "non_success"
    assert ci.show(1)["failures"][0]["job"] == "<workflow>"


def test_unknown_state_version_is_not_overwritten(ci: CIEnv) -> None:
    assert ci.start().returncode == 0
    with sqlite3.connect(ci.local / "observations.sqlite3") as db:
        db.execute("PRAGMA user_version = 9")
    assert ci.collect().returncode == 1
    with sqlite3.connect(ci.local / "observations.sqlite3") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 9


def test_bounded_local_reads_evidence_integrity_and_no_writes(ci: CIEnv) -> None:
    ci.run_fixture(1, "failure")
    assert ci.start().returncode == 0
    calls = ci.calls.read_text()
    page1 = ci.records("--limit", "1")
    assert page1["has_more"] is True
    page2 = ci.records("--after", str(page1["next_after"]), "--limit", "1")
    assert page2["next_after"] > page1["next_after"]
    failure = ci.show(1)["failures"][0]
    source = failure["evidence"][0]["source"]
    result = ci.cli(
        "evidence", "--object", source["object"], "--start-line", "1", "--limit", "1"
    )
    assert result.returncode == 0
    assert "FAILED tests/test_scene.py" in json.loads(result.stdout)["lines"][0]
    assert ci.calls.read_text() == calls
    assert all(
        json.loads(line)[:5] == ["api", "--hostname", "github.com", "--method", "GET"]
        for line in calls.splitlines()
    )
    assert not (ROOT / ".agents/ops/ci_watch/systemd").exists()
    (ci.local / "objects" / source["object"]).write_text("corrupt")
    assert ci.cli("evidence", "--object", source["object"]).returncode == 1


def test_explicit_bootstrap_config_binding_and_writer_lock(ci: CIEnv) -> None:
    assert ci.collect().returncode == 1
    assert not ci.calls.exists()
    assert ci.start().returncode == 0
    calls = ci.calls.read_text()
    assert ci.collect("--start-sha", sha(2)).returncode == 1
    assert ci.collect("--workflow", "other.yml").returncode == 1
    assert ci.cli("collect", "--repo", "o/else").returncode == 1
    with (ci.local / "collect.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ci.collect().returncode == 1
    assert ci.calls.read_text() == calls
