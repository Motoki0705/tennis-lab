"""Bounded GitHub reads across the real dispatcher worker process boundary."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_supervisor import d as d
from tests.e2e.agents_ops.test_dispatcher_supervisor import (
    registered_repair as registered_repair,
)


def progression(d: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict, Path]:
    d.publish_github(monkeypatch)
    endpoints = d.h.root / "process-http.jsonl"
    monkeypatch.setenv("FAKE_DISPATCH_HTTP_CALLS", str(endpoints))
    row = d.row()
    saved = row["repair_progressions"]["pipeline-1"]
    state = importlib.import_module("lib.ops_review.progression").new_state(
        {
            "schema_version": 4,
            "review_run_id": "parent-1-" + "a" * 32,
            "repo": "o/r",
            "parent_issue": 1,
            "authorization": saved["authorization"],
            "prs": [
                {
                    "number": 42,
                    "issues": [10],
                    "worker": {
                        "kind": "dispatcher",
                        "child": 10,
                        "delegation_id": saved["delegation_id"],
                        "progression_id": "pipeline-1",
                    },
                }
            ],
        },
        owner=row["session"],
        generation=row["generation"],
    )
    state["options"] = {"dispatcher_state_dir": str(d.schedule.directory)}
    backend = importlib.import_module("lib.ops_review.progression_backend").LiveBackend(
        state, d.h.store
    )
    return backend, state, endpoints


def fake_runner(d: Any) -> Path:
    """Replace only the final runner; worker.py and its admission remain real."""
    runner = Path(d.h.root) / "process-runner"
    runner.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "root=Path(args[args.index('--run-dir')+1]).parent\n"
        "(root/'launched.json').write_text(json.dumps(args))\n"
    )
    runner.chmod(0o700)
    with d.h.store.locked():
        saved = d.schedule.read()
        saved["config"]["runner"] = str(runner)
        d.schedule.write(saved)
    return runner


def spawn(
    d: Any, runtime_type: Any, job: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wait on the actual child handle, including admission rejection before launch."""
    processes = []
    original = subprocess.Popen

    def capture(*args: Any, **kwargs: Any) -> Any:
        child = original(*args, **kwargs)
        processes.append(child)
        return child

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", capture)
        runtime_type(Path(job["runner"])).spawn(
            d.h.store.directory,
            "o/r#1",
            job["id"],
            Path(job["directory"]),
            schedule_dir=d.schedule.directory,
        )
    assert len(processes) == 1
    try:
        code = processes[0].wait(timeout=30)
    finally:
        if processes[0].poll() is None:
            processes[0].terminate()
            processes[0].wait(timeout=10)
    assert code == 0, (Path(job["directory"]) / "worker.log").read_text()


def add_unrelated(d: Any, count: int) -> int:
    path = d.h.root / "github.json"
    remote = json.loads(path.read_text())
    evidence = next(
        row
        for row in remote["issues"].values()
        if row.get("labels") == [{"name": "parent-evidence"}]
    )
    body = importlib.import_module("parent.evidence").issue_body
    for i in range(count):
        n = 80000 + i
        remote["issues"][str(n)] = {
            **copy.deepcopy(evidence),
            "number": n,
            "id": n,
            "body": body("o/r", 90000 + i),
            "html_url": f"https://github.com/o/r/issues/{n}",
        }
    path.write_text(json.dumps(remote))
    return int(evidence["number"])


def reads(path: Path) -> Counter:
    endpoints = [json.loads(line) for line in path.read_text().splitlines()]
    assert endpoints
    assert not any(
        e.partition("?")[0] in {"repos/o/r/issues", "repos/o/r/pulls"}
        for e in endpoints
    ), endpoints
    unrelated = {str(n) for n in range(80000, 80160)} | {
        str(n) for n in range(90000, 90160)
    }
    assert not any(
        e.partition("?")[0].split("/")[4] in unrelated
        for e in endpoints
        if e.startswith(("repos/o/r/issues/", "repos/o/r/pulls/"))
    )
    return Counter(endpoints)


@pytest.mark.parametrize("fault", [None, "missing_binding", "unavailable_binding"])
def test_progression_dispatcher_adapter_starts_repair_without_repository_listing(
    registered_repair: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    fault: str | None,
) -> None:
    d, runtime_type = registered_repair
    backend, state, endpoints = progression(d, monkeypatch)
    fake_runner(d)
    counts = []
    for unrelated in (0, 160) if fault is None else (160,):
        evidence = add_unrelated(d, unrelated)
        attempt = {
            "id": f"repair-process-{unrelated}",
            "details": {"reason": "ci_failure"},
        }
        assert backend.repair(state, state["targets"]["42"], attempt) == {
            "outcome": "running"
        }
        job = next(
            j
            for j in d.row()["dispatch"]["jobs"].values()
            if j.get("repair", {}).get("repair_id") == attempt["id"]
        )
        if fault == "missing_binding":
            d.h.gh.evidence_issues().path(1).unlink()
        elif fault == "unavailable_binding":
            monkeypatch.setenv("FAKE_DISPATCH_FAIL_GET", f"repos/o/r/issues/{evidence}")
        endpoints.write_text("")
        spawn(d, runtime_type, job, monkeypatch)
        calls = reads(endpoints)
        current = d.row()["dispatch"]["jobs"][job["id"]]
        launched = Path(job["directory"]) / "launched.json"
        if fault is not None:
            assert current["cancelled_before_start"] is True
            assert (
                "saved evidence issue binding" if fault == "missing_binding" else "503"
            ) in current["diagnostic"]
            assert not launched.exists()
            return
        assert current["runner_exit"] == 0
        argv = json.loads(launched.read_text())
        assert argv[0] == "resume" and argv[argv.index("--effort") + 1] == "high"
        assert argv[argv.index("--resume-from") + 1] == job["resume_from"]
        counts.append(calls)
        # Fake the common runner's terminal artifacts, then consume them through
        # the real supervisor/progression. No in-process claim replaces worker.py.
        d.runtime(job)
        record_path = Path(job["directory"]) / "runtime/result.json"
        record = json.loads(record_path.read_text())
        settings = importlib.import_module("lib.agent_settings")
        record["contract"]["settings"] = settings.resolve(
            settings.DEFAULT_CONFIG, "codex", None, "gpt-6-astra", None, "write"
        )
        record["contract"]["settings"]["selected"]["effort"] = "high"
        record["contract"]["repair"] = job["repair"]
        record["contract"]["branch"] = job["branch"]
        record_path.write_text(json.dumps(record))
        report = {
            "schema_version": 1,
            "repair_id": attempt["id"],
            "pr": 42,
            "outcome": "repaired",
            "head_sha": "b" * 40,
            "summary": "Fixed import",
            "verification": ["fixture check"],
            "parent_request": None,
        }
        (record_path.parent / "final.md").write_text(json.dumps(report))
        remote_path = d.h.root / "github.json"
        remote = json.loads(remote_path.read_text())
        remote["prs"][0]["head"]["sha"] = report["head_sha"]
        remote_path.write_text(json.dumps(remote))
        backend.dispatcher().reconcile()
        assert backend.recover_repair(state, state["targets"]["42"], attempt) == report
    assert counts[0] == counts[1]
    request.node.user_properties.append(
        ("worker_get_counts_0_160", [sum(c.values()) for c in counts])
    )


def test_progression_notice_resumes_parent_with_bounded_process_reads(
    registered_repair: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d, runtime_type = registered_repair
    row = d.row()
    row["dispatch"].update(phase="waiting", wake_requested=False)
    d.update(row)
    d.runtime(d.job(row), state="waiting")
    d.control.reconcile()
    backend, state, endpoints = progression(d, monkeypatch)
    fake_runner(d)
    add_unrelated(d, 160)
    GitHub = importlib.import_module("parent.github").GitHub
    control = d.mod.Supervisor(GitHub("o/r", store=d.h.store), d.h.store, d.schedule)
    backend.notify(state, {"id": 1, "kind": "completed", "reason": None})
    # The normal supervisor after a process restart must honor the saved notice.
    control.reconcile()
    job = d.job()
    assert job["role"] == "parent" and job["phase"] == "reserved"
    spawn(d, runtime_type, job, monkeypatch)
    argv = json.loads((Path(job["directory"]) / "launched.json").read_text())
    assert argv[0] == "resume" and argv[argv.index("--role") + 1] == "parent"
    reads(endpoints)
