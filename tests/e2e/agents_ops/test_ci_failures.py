"""Real observer and repair CLIs, durable evidence, fake GitHub, zero agents."""

from __future__ import annotations

import fcntl
import importlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
sys.path.insert(0, str(OPS / "ops-report"))
issues: Any = importlib.import_module("issues")
spec = importlib.util.spec_from_file_location(
    "ci_observer_fixture", Path(__file__).with_name("test_ci_watch.py")
)
assert spec is not None and spec.loader is not None
observer: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
FAKE = Path(__file__).with_name("fake_ci_failure_gh.py")
PASSED = "2026-10-24T02:00:00Z PASSED tests/test_scene.py::test_tracks\n"


class RepairEnv:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.ci = observer.CIEnv(tmp)
        self.remote = tmp / "repairs.json"
        self.calls = tmp / "repair-calls.jsonl"
        self.local = tmp / "repairs"
        self.data: dict[str, Any] = {
            "issues": [],
            "pulls": [],
            "comments": {},
            "next_number": 1,
        }
        self.save()
        gh = tmp / "bin/gh"
        gh.write_text(
            f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(FAKE)!r}, run_name='__main__')\n"
        )
        for executable in ("codex", "claude", "systemctl", "agent_exec.sh"):
            trap = tmp / "bin" / executable
            trap.write_text(
                f"#!/bin/sh\ntouch '{tmp / 'FORBIDDEN-EXECUTION'}'\nexit 99\n"
            )
            trap.chmod(0o700)
        self.env = {
            **self.ci.env,
            "FAKE_FAILURE_STATE": str(self.remote),
            "FAKE_FAILURE_CALLS": str(self.calls),
        }

    def save(self) -> None:
        self.remote.write_text(json.dumps(self.data))

    def load(self) -> dict[str, Any]:
        self.data = json.loads(self.remote.read_text())
        return self.data

    def cli(self, *args: str, **changes: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ci_failures",
                "--state-dir",
                str(self.local),
                *args,
            ],
            cwd=ROOT,
            env={**self.env, **changes},
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert not (self.tmp / "FORBIDDEN-EXECUTION").exists()
        return result

    def sync(self, **changes: str) -> subprocess.CompletedProcess[str]:
        return self.cli(
            "sync",
            "--repo",
            "o/r",
            "--source-dir",
            str(self.ci.local),
            "--apply",
            **changes,
        )

    def collect(self, *, first: bool = False) -> None:
        result = self.ci.start() if first else self.ci.collect()
        assert result.returncode == 0, result.stderr

    def regression(self) -> str:
        self.ci.run_fixture(1, "failure")
        self.ci.commit(2, 1)
        self.ci.run_fixture(2, "failure")
        self.collect(first=True)
        return str(self.ci.show(2)["failures"][0]["signature"])

    def show(self, signature: str) -> dict[str, Any]:
        result = self.cli("show", "--signature", signature)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["record"]  # type: ignore[no-any-return]

    def issue(self, number: int, signature: str, **changes: Any) -> None:
        self.data["issues"].append(
            {
                "number": number,
                "id": number + 10000,
                "title": "既存の改名済みIssue",
                "body": issues.ci_failure_marker(signature)
                + "\n\n既存の要件を維持する。",
                "repository_url": "https://api.github.com/repos/o/r",
                "state": "open",
                "state_reason": None,
                "labels": [],
                "assignees": [],
                **changes,
            }
        )
        self.data["next_number"] = max(self.data["next_number"], number + 1)
        self.save()

    def pr(
        self, *, signature: str | None = None, merged: bool = False, **changes: Any
    ) -> None:
        self.load()
        self.data["pulls"] = [
            {
                "number": 80,
                "state": "closed" if merged else "open",
                "merged_at": "2026-10-24T02:00:00Z" if merged else None,
                "merge_commit_sha": observer.sha(3) if merged else None,
                "body": issues.ci_failure_marker(signature) + "\n\n修正する"
                if signature
                else "Closes #1",
                "head": {"sha": "a" * 40, "ref": "fix-ci"},
                "base": {"ref": "main"},
                **changes,
            }
        ]
        self.save()

    def success(self, commit: int, *, log: str = PASSED) -> None:
        self.ci.commit(commit, commit - 1)
        self.ci.run_fixture(commit, log=log)
        self.collect()

    def assert_only_supported_writes(self) -> None:
        for call in map(json.loads, self.calls.read_text().splitlines()):
            assert call[0] == "api"
            method = call[call.index("--method") + 1]
            endpoint = next(a for a in call if a == "user" or a.startswith("repos/"))
            if method != "GET":
                assert method == "POST"
                assert endpoint == "repos/o/r/issues" or endpoint.endswith("/comments")


@pytest.fixture
def repair(tmp_path: Path) -> RepairEnv:
    return RepairEnv(tmp_path)


def test_new_regression_idempotence_new_sha_and_dispatcher_collection(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    first = repair.sync()
    assert first.returncode == 0, first.stderr
    data = repair.load()
    assert len(data["issues"]) == 1
    created = data["issues"][0]
    assert created["body"].splitlines()[0] == issues.ci_failure_marker(signature)
    assert created["labels"] == [{"name": "bug"}, {"name": "ai-proposed"}]
    assert "ops-report-proposal" not in created["body"]
    assert (
        issues.validate_acceptance(
            issues.parse_sections(created["body"])["Acceptance checklist"]
        )
        is None
    )
    before = repair.cli("records").stdout
    assert repair.sync().returncode == 0
    assert repair.cli("records").stdout == before
    assert repair.load() == data
    repair.ci.commit(3, 2)
    repair.ci.run_fixture(3, "failure")
    repair.collect()
    assert repair.sync().returncode == 0
    assert len(repair.load()["issues"]) == 1
    comments = repair.data["comments"]["1"]
    assert comments[:2] == data["comments"]["1"]
    assert sum(f"SHA {observer.sha(3)}" in c["body"] for c in comments) == 1
    work = repair.tmp / "code"
    work.mkdir()
    for args in (
        ("init", "-q"),
        (
            "-c",
            "user.name=Fake",
            "-c",
            "user.email=fake@example.org",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ),
    ):
        subprocess.run(["git", *args], cwd=work, check=True, capture_output=True)
    program = (
        "from pathlib import Path\nfrom dispatcher.collect import collect\n"
        "from parent.github import GitHub\nfrom parent.state import Store\n"
        f"store = Store(Path({str(repair.tmp / 'parents')!r}))\n"
        f"with store.locked():\n snapshot = collect(GitHub('o/r'), store, Path({str(work)!r}))\n"
        "assert snapshot['subjects'] == [1]\nassert snapshot['issues']['1']['parent'] is None\n"
        "assert snapshot['issues']['1']['source_reports'] == []\nprint('collected')\n"
    )
    collected = subprocess.run(
        [sys.executable, "-c", program],
        env=repair.env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert collected.returncode == 0, collected.stderr
    repair.assert_only_supported_writes()


@pytest.mark.parametrize("state", ["open", "closed"])
def test_reuses_all_issue_states_preserving_existing_work(
    repair: RepairEnv, state: str
) -> None:
    signature = repair.regression()
    repair.issue(
        45,
        signature,
        state=state,
        state_reason="completed" if state == "closed" else None,
        assignees=[{"login": "owner"}],
    )
    original = repair.data["issues"][0].copy()
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert repair.load()["issues"] == [original]
    assert len(repair.data["comments"]["45"]) == 2
    row = repair.show(signature)
    assert row["issue"] == 45 and row["status"] == "awaiting_fix"


@pytest.mark.parametrize("merged", [False, True])
def test_existing_pr_with_source_prevents_new_issue(
    repair: RepairEnv, merged: bool
) -> None:
    signature = repair.regression()
    repair.pr(signature=signature, merged=merged)
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert repair.load()["issues"] == []
    assert len(repair.data["comments"]["80"]) == 2
    assert repair.show(signature)["prs"] == [80]


@pytest.mark.parametrize(
    "lost_path", ["repos/o/r/issues", "repos/o/r/issues/1/comments"]
)
def test_lost_creation_or_comment_response_resumes_without_duplicate(
    repair: RepairEnv, lost_path: str
) -> None:
    signature = repair.regression()
    lost = repair.sync(FAKE_FAILURE_LOSE_RESPONSE=lost_path)
    assert lost.returncode == 1 and "lost response" in lost.stderr
    assert len(repair.load()["issues"]) == 1
    assert repair.show(signature)["occurrences"]
    assert repair.show(signature)["status"] == "publication_pending"
    resumed = repair.sync()
    assert resumed.returncode == 0, resumed.stderr
    assert len(repair.load()["issues"]) == 1
    assert len(repair.data["comments"]["1"]) == 2


@pytest.mark.parametrize("category", ["flaky", "runner_infrastructure", "unclassified"])
def test_non_code_categories_remain_evidenced_without_repair_issue(
    repair: RepairEnv, category: str
) -> None:
    if category == "runner_infrastructure":
        repair.ci.run_fixture(
            1,
            "failure",
            log="runner has lost communication with the server\n",
            step="Setup",
        )
    else:
        repair.ci.run_fixture(1, "failure")
        if category == "flaky":
            repair.ci.run_fixture(1, attempt=2)
    repair.collect(first=True)
    signature = repair.ci.show(1)["failures"][0]["signature"]
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert repair.load()["issues"] == [] and repair.data["comments"] == {}
    row = repair.show(signature)
    assert row["status"] == "held"
    assert row["occurrences"][0]["failure"]["classification"]["category"] == category
    assert row["occurrences"][0]["evidence"]["target_version"] == observer.sha(1)


def test_recovery_requires_merge_success_and_exact_test_and_can_regress(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.success(3)
    assert repair.sync().returncode == 0
    assert repair.show(signature)["status"] == "awaiting_fix"
    repair.pr(merged=True)
    assert repair.sync().returncode == 0
    recovered = repair.show(signature)
    assert recovered["status"] == "recovered"
    assert recovered["recovery"]["sha"] == observer.sha(3)
    assert recovered["recovery"]["pr"] == 80
    assert recovered["recovery"]["jobs"][0]["job_url"].endswith("/300001")
    snapshot = repair.cli("records").stdout
    assert repair.sync().returncode == 0
    assert repair.cli("records").stdout == snapshot
    assert len(repair.load()["comments"]["1"]) == 3
    summary = json.loads(repair.cli("list", "--unresolved").stdout)
    assert summary["total"] == 0
    repair.ci.commit(4, 3)
    repair.ci.run_fixture(4, "failure")
    repair.collect()
    assert repair.sync().returncode == 0
    row = repair.show(signature)
    assert row["status"] == "regressed" and row["recovery"] is None
    assert len(repair.load()["issues"]) == 1
    repair.success(5)
    assert repair.sync().returncode == 0
    assert repair.show(signature)["recovery"]["sha"] == observer.sha(5)


def test_interrupted_collection_fetch_error_cannot_restore_old_recovery(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.pr(merged=True)
    repair.success(3)
    assert repair.sync().returncode == 0
    repair.ci.commit(4, 3)
    repair.ci.run_fixture(4, "failure")
    repair.collect()
    assert repair.sync().returncode == 0
    assert repair.show(signature)["status"] == "regressed"

    # The collector can die after publishing a SHA but before its collection
    # summary. Keep the old error-free summary and append the new failed read.
    source = importlib.import_module("ci_watch.store").Store(repair.ci.local)
    old = source.records(0, 1000, observer.sha(4))["records"][-1]
    failed = {
        **old["observation"],
        "status": "fetch_error",
        "result": None,
        "failures": [],
        "workflows": [],
        "errors": [{"endpoint": "runs", "reason": "temporary outage"}],
    }
    with source.db:
        source.append(
            "observation", observer.sha(4), failed, datetime.now(UTC).isoformat()
        )
    source.close()
    result = repair.sync()
    assert result.returncode == 1, result.stderr
    row = repair.show(signature)
    assert row["status"] == "observation_incomplete"
    assert row["recovery"] is None
    assert any(o["failure"]["sha"] == observer.sha(4) for o in row["occurrences"])
    assert not json.loads(result.stdout)["processed"]


def test_successful_retry_updates_recovery_evidence_idempotently(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.pr(merged=True)
    repair.success(3)
    assert repair.sync().returncode == 0
    repair.ci.run_fixture(3, attempt=2, log=PASSED)
    repair.collect()
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    row = repair.show(signature)
    assert row["status"] == "recovered"
    assert row["recovery"]["jobs"][0]["attempt"] == 2
    comments = repair.load()["comments"]["1"]
    recovered = [c for c in comments if "ci-failure-recovery:" in c["body"]]
    assert len(recovered) == 2
    assert recovered[0]["body"].splitlines()[0] != recovered[1]["body"].splitlines()[0]
    assert repair.sync().returncode == 0
    assert repair.load()["comments"]["1"] == comments


@pytest.mark.parametrize(
    "kind",
    [
        "no_test_proof",
        "other_test",
        "skipped_step",
        "other_job",
        "wrong_merge_sha",
        "other_base",
        "closed_only",
        "post_merge_failure",
    ],
)
def test_unverified_or_failed_ci_never_counts_as_recovery(
    repair: RepairEnv, kind: str
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.pr(merged=True)
    if kind == "wrong_merge_sha":
        repair.pr(merged=True, merge_commit_sha="f" * 40)
    elif kind == "other_base":
        repair.pr(merged=True, base={"ref": "release"})
    elif kind == "closed_only":
        repair.load()
        repair.data["pulls"] = []
        repair.data["issues"][0].update(state="closed", state_reason="completed")
        repair.save()
    repair.ci.commit(3, 2)
    log = (
        "1 passed\n"
        if kind == "no_test_proof"
        else "PASSED tests/test_scene.py::test_tracks_extra\n"
        if kind == "other_test"
        else PASSED
    )
    repair.ci.run_fixture(
        3,
        "failure" if kind == "post_merge_failure" else "success",
        log=observer.TEST_LOG if kind == "post_merge_failure" else log,
    )
    if kind == "skipped_step":
        repair.ci.state["jobs"]["300/1"][0]["steps"][0]["conclusion"] = "skipped"
    elif kind == "other_job":
        repair.ci.state["jobs"]["300/1"][0]["name"] = "Renamed Python"
    repair.collect()
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    row = repair.show(signature)
    assert row["status"] != "recovered" and row["recovery"] is None
    assert json.loads(repair.cli("list", "--unresolved").stdout)["total"] == 1


def test_repository_lock_spans_distinct_state_directories(repair: RepairEnv) -> None:
    repair.regression()
    lock_path = repair.tmp / "xdg/tennis-lab-agents/ci-failure-locks/o--r.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = repair.sync()
    assert result.returncode == 1 and "repository lock" in result.stderr
    assert not repair.calls.exists()
    assert not repair.local.exists()


def test_concurrent_sync_cannot_create_twice(repair: RepairEnv) -> None:
    repair.regression()
    ready = repair.tmp / "ready"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ci_failures",
            "--state-dir",
            str(repair.local),
            "sync",
            "--repo",
            "o/r",
            "--source-dir",
            str(repair.ci.local),
            "--apply",
        ],
        env={**repair.env, "FAKE_FAILURE_CREATE_PAUSE": str(ready)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert ready.exists()
        repair.local = repair.tmp / "another-state"
        conflict = repair.sync()
        assert conflict.returncode == 1 and "repository lock" in conflict.stderr
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        assert len(repair.load()["issues"]) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_all_pages_and_duplicate_or_malformed_source_fail_closed(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    for n in range(1, 102):
        repair.issue(n, signature, body="unrelated")
    repair.issue(150, signature, state="closed")
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert len(repair.load()["issues"]) == 102
    assert repair.show(signature)["issue"] == 150
    repair.issue(151, signature)
    before = repair.load()
    assert "duplicate source" in repair.sync().stderr
    assert repair.load() == before


def test_immutable_local_references_and_ledger_evidence(repair: RepairEnv) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    event = json.loads(repair.cli("show", "--signature", signature).stdout)
    ref = repair.tmp / "reference.json"
    ref.write_text(json.dumps(event["ref"]))
    repair.pr(merged=True)
    repair.success(3)
    assert repair.sync().returncode == 0
    original = repair.cli("original", "--reference", str(ref))
    assert original.returncode == 0, original.stderr
    assert json.loads(original.stdout) == event
    row = repair.show(signature)
    ledger = importlib.import_module("opsreport.ledger")
    at = datetime.now(UTC).isoformat()
    book = ledger.Ledger.create(
        repair.tmp / "ledger.json",
        slot="20261024",
        base_sha=observer.sha(2),
        goal="CI failures",
        at=at,
    )
    records = [o["evidence"] for o in row["occurrences"]] + [
        row["recovery"]["evidence"]
    ]
    for item in {r["id"]: r for r in records}.values():
        book.record_evidence(item, at=at)
    assert len(book.snapshot["evidence"]) == 3
    obj = row["recovery"]["jobs"][0]["test_log_object"]
    result = repair.cli("evidence", "--object", obj, "--limit", "1")
    assert json.loads(result.stdout)["lines"] == PASSED.splitlines()
    reference = event["ref"].copy()
    reference["sha256"] = "f" * 64
    ref.write_text(json.dumps(reference))
    assert repair.cli("original", "--reference", str(ref)).returncode == 1


def test_source_metadata_uses_shared_provenance_contract() -> None:
    signature = "ci-failure-v1:" + "a" * 64
    marker = issues.ci_failure_marker(signature)
    assert issues.index_ci_failures([(1, marker + "\nbody")]) == {signature: 1}
    assert issues.parse_ci_failure_source("```\n" + marker + "\n```") is None
    for body in ("text\n" + marker, marker + "\n" + marker, marker.replace("v1", "v2")):
        with pytest.raises(ValueError):
            issues.parse_ci_failure_source(body)


def test_existing_issue_and_closing_pr_both_receive_evidence(repair: RepairEnv) -> None:
    signature = repair.regression()
    repair.issue(1, signature)
    repair.pr()
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert len(repair.load()["issues"]) == 1
    assert len(repair.data["comments"]["1"]) == len(repair.data["comments"]["80"]) == 2


def test_classification_reversal_does_not_open_another_issue_or_confirm_recovery(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.ci.run_fixture(2, attempt=2)
    repair.collect()
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    assert len(repair.load()["issues"]) == 1
    row = repair.show(signature)
    assert row["status"] != "recovered"
    assert {o["failure"]["classification"]["category"] for o in row["occurrences"]} == {
        "flaky",
        "unclassified",
    }


def test_job_level_recovery_does_not_claim_individual_test_execution(
    repair: RepairEnv,
) -> None:
    log = "tests/helper.py:12: ValueError: incompatible input\n"
    repair.ci.run_fixture(1, "failure", log=log)
    repair.ci.commit(2, 1)
    repair.ci.run_fixture(2, "failure", log=log)
    repair.collect(first=True)
    signature = repair.ci.show(2)["failures"][0]["signature"]
    assert repair.sync().returncode == 0
    repair.pr(merged=True)
    repair.success(3, log="job completed\n")
    result = repair.sync()
    assert result.returncode == 0, result.stderr
    row = repair.show(signature)
    assert row["status"] == "recovered"
    assert row["occurrences"][0]["failure"]["test_id"] is None
    assert row["recovery"]["jobs"][0]["test_log_object"] is None


def test_observer_lock_and_collection_failure_are_explicit(repair: RepairEnv) -> None:
    signature = repair.regression()
    with (repair.ci.local / "collect.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = repair.sync()
    assert result.returncode == 1 and "collector is active" in result.stderr
    assert not repair.calls.exists()
    assert repair.sync().returncode == 0
    repair.ci.state["errors"]["repos/o/r/git/ref/heads/main"] = "403 forbidden"
    failed = repair.ci.collect()
    assert failed.returncode == 1
    result = repair.sync()
    assert result.returncode == 1
    assert json.loads(result.stdout)["errors"]
    assert repair.show(signature)["status"] == "observation_incomplete"


def test_success_log_fetch_failure_and_cached_proof_are_not_hidden(
    repair: RepairEnv,
) -> None:
    signature = repair.regression()
    assert repair.sync().returncode == 0
    repair.pr(merged=True)
    repair.success(3)
    repair.load()
    repair.data["error"] = "repos/o/r/actions/jobs/300001/logs"
    repair.save()
    result = repair.sync()
    assert result.returncode == 1 and "retrieval failure" in result.stderr
    assert repair.show(signature)["status"] != "recovered"
    repair.load()
    del repair.data["error"]
    repair.save()
    assert repair.sync().returncode == 0
    obj = repair.show(signature)["recovery"]["jobs"][0]["test_log_object"]
    repair.load()
    repair.data["error"] = "repos/o/r/actions/jobs/300001/logs"
    repair.save()
    assert repair.sync().returncode == 0  # Completed job's verified local original.
    (repair.local / "objects" / obj).write_text("corrupted")
    result = repair.sync()
    assert result.returncode == 1 and "content mismatch" in result.stderr


def test_database_corruption_is_not_reinitialized(repair: RepairEnv) -> None:
    repair.regression()
    assert repair.sync().returncode == 0
    with sqlite3.connect(repair.local / "failures.sqlite3") as db:
        db.execute("DROP TABLE events")
    result = repair.sync()
    assert result.returncode == 1 and "schema mismatch" in result.stderr
    with sqlite3.connect(repair.local / "failures.sqlite3") as db:
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='events'"
        ).fetchone()
