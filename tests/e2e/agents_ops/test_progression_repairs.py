"""The progression consumes #2175's actual manual resume and dispatcher notices."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import latest, store, task
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_repair_workers import creating_worker


@pytest.mark.parametrize("outcome", ["repaired", "needs_parent"])
def test_live_adapter_resumes_creating_session_and_consumes_strict_report(
    reserved: ParentEnv, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    authorization = importlib.import_module("parent.execution").authorization
    GitHub = importlib.import_module("parent.github").GitHub
    ParentSession = importlib.import_module("parent.session").ParentSession
    progression = importlib.import_module("lib.ops_review.progression")
    new_state, save = progression.new_state, progression.save
    LiveBackend = importlib.import_module(
        "lib.ops_review.progression_backend"
    ).LiveBackend

    p = reserved
    first = creating_worker(p)
    before = json.loads((first / "result.json").read_text())
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(p.repo)
    monkeypatch.setenv("AGENT_AUTO_MERGE", "1")
    api = importlib.import_module("parent.manual_workers")
    monkeypatch.setattr(api, "RUNNER", p.root / "fake-runner")
    registry = store(p)
    monkeypatch.setenv("MANUAL_STATE_DIR", str(registry.directory))
    monkeypatch.setenv("MANUAL_RUNNER_CALLS", str(p.root / "runner-calls.jsonl"))
    monkeypatch.setenv("MANUAL_RUNNER_PROMPTS", str(p.root / "runner-prompts.jsonl"))
    identity = "claude:" + p.env["CLAUDE_CODE_SESSION_ID"]
    with registry.locked():
        session = ParentSession(GitHub("o/r", store=registry), registry, 1, identity)
        auth = authorization(session)
    request = {
        "schema_version": 4,
        "review_run_id": "parent-1-" + "1" * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "authorization": auth,
        "prs": [
            {
                "number": 42,
                "issues": [10],
                "worker": {
                    "kind": "manual",
                    "child": 10,
                    "delegation_id": task(p)["delegation_id"],
                },
            }
        ],
    }
    state = new_state(request, owner=identity, generation=auth["generation"])
    state.update(checkout=str(p.repo), options={})
    path = registry.directory / "progressions" / state["id"] / "state.json"
    path.parent.mkdir(parents=True)
    save(path, state)
    head = p.state()["prs"][0]["head"]["sha"]
    report = {
        "schema_version": 1,
        "repair_id": "repair-1",
        "pr": 42,
        "outcome": outcome,
        "head_sha": head,
        "summary": "Fixed test import" if outcome == "repaired" else "Needs a decision",
        "verification": ["pytest passed"],
        "parent_request": None
        if outcome == "repaired"
        else "Which behavior is intended?",
    }
    monkeypatch.setenv("MANUAL_PLANNING_FINAL", json.dumps(report))
    backend = LiveBackend(state, registry)
    attempt = {
        "id": "repair-1",
        "details": {"reason": "ci_failure", "job": "tests", "log": "import failed"},
    }
    before_reads = len(p.calls())
    returned = backend.repair(state, state["targets"]["42"], attempt)
    assert returned == report
    assert backend.recover_repair(state, state["targets"]["42"], attempt) == report
    after = json.loads((latest(p) / "result.json").read_text())
    assert after["runtime"]["session_id"] == before["runtime"]["session_id"]
    assert (
        after["contract"]["settings"]["selected"]["model"]
        == before["contract"]["settings"]["selected"]["model"]
    )
    assert after["contract"]["settings"]["selected"]["effort"] == "high"
    assert len((p.root / "runner-calls.jsonl").read_text().splitlines()) == 2
    assert not any(
        arg.partition("?")[0] in {"repos/o/r/issues", "repos/o/r/pulls"}
        for call in p.calls()[before_reads:]
        for arg in call
    )


def test_repair_failure_details_include_attempt_and_unavailable_log_reason(
    tmp_path: Path,
) -> None:
    Store = importlib.import_module("parent.state").Store
    LiveBackend = importlib.import_module(
        "lib.ops_review.progression_backend"
    ).LiveBackend

    state = {"request": {"repo": "o/r", "parent_issue": 1}}
    backend = LiveBackend(state, Store(tmp_path))

    class Fake:
        def get(self, endpoint: str) -> dict:
            if "/actions/runs?" in endpoint:
                return {
                    "total_count": 1,
                    "workflow_runs": [
                        {"id": 50, "check_suite_id": 2, "run_attempt": 3}
                    ],
                }
            assert "/runs/50/attempts/3/jobs?" in endpoint
            return {
                "total_count": 1,
                "jobs": [
                    {
                        "id": 60,
                        "name": "pytest",
                        "conclusion": "failure",
                        "html_url": "https://github.com/o/r/actions/runs/50/job/60",
                        "steps": [{"name": "test", "conclusion": "failure"}],
                    }
                ],
            }

        def job_log(self, job_id: int, **kwargs):
            JobLog = importlib.import_module("issues.polling").JobLog

            assert job_id == 60
            return JobLog(404, None, "log unavailable")

    backend.gh = Fake()
    details = backend.failure(
        {"number": 11},
        {"head": "a" * 40, "checks": [{"name": "Python", "check_suite": {"id": 2}}]},
    )
    (job,) = details["failed_jobs"]
    assert job["attempt"] == 3 and job["steps"][0]["name"] == "test"
    assert job["log"] is None and job["log_unavailable_reason"] == "log unavailable"
