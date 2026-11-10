"""Durable private storage, real concurrent writers and publication recovery."""

from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from tests.support.agents_ops.decision_logs import (
    AT,
    PublicationError,
    decision,
    evidence,
    history,
    journal,
    result,
    schema,
)


def test_capture_replay_and_publication_are_independent_and_private(
    tmp_path: Path,
) -> None:
    store = journal.Journal(tmp_path / "state", "o/r", 1)
    assert store.status() == [] and not store.path.exists()
    raw = result()
    del raw["id"]
    raw["data"]["summary"] = "private-session-value saved /home/private/report"
    private = {
        "sessions": ["private-session-value"],
        "runs": ["private-run-value"],
        "identifiers": [],
        "roots": [],
        "checkout_commit": None,
    }
    record = store.capture("private-delivery-key", raw, private=private)
    first_bytes = store.path.read_bytes()
    replay = journal.Journal(tmp_path / "state", "o/r", 1).capture(
        "private-delivery-key", raw, private=private
    )
    assert replay["id"] == record["id"] and store.path.read_bytes() == first_bytes
    assert store.status()[0]["publication_status"] == "unprepared"
    prefix = store.read()["entries"]
    prepared = store.prepare(record["id"], key="prepare-1", at=AT)
    assert prepared["status"] == "ready"
    assert store.status()[0]["capture_status"] == "captured"
    assert store.status()[0]["publication_status"] == "ready"
    store.publication(
        record["id"], key="git-start", status="pending", at=AT, evidence=[]
    )
    assert store.status()[0]["publication_status"] == "pending"
    store.publication(
        record["id"],
        key="git-failed",
        status="blocked",
        at=AT,
        evidence=[],
        error="github_failed",
    )
    assert store.status()[0]["publication_status"] == "blocked"
    assert store.read()["entries"][:1] == prefix
    store.publication(
        record["id"], key="git-retry", status="pending", at=AT, evidence=[]
    )
    store.publication(
        record["id"], key="git-saved", status="published", at=AT, evidence=[evidence()]
    )
    saved_bytes = store.path.read_bytes()
    store.publication(
        record["id"], key="git-saved", status="published", at=AT, evidence=[evidence()]
    )
    assert store.path.read_bytes() == saved_bytes
    public = json.dumps(store.public_records())
    assert "private-session-value" not in public and "/home/private" not in public
    assert "private-delivery-key" not in public and "private-run-value" not in public
    assert "[private identifier]" in public and "[local path]" in public
    assert store.public_payloads()[0]["sanitization"] == {
        "version": 2,
        "replacements": 2,
        "local_paths": {"posix": 1},
    }
    assert store.status()[0]["result"] == {
        "decision": "undecided",
        "parent_confirmation": "unrecorded",
        "final_report": "unavailable",
    }
    assert store.status()[0]["publication_status"] == "published"
    assert (
        store.read()["entries"][0]["record"]["data"]["summary"]
        == raw["data"]["summary"]
    )
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.with_suffix(".lock").stat().st_mode) == 0o600
    for directory in (
        store.path.parent,
        store.path.parent.parent,
        store.path.parent.parent.parent,
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_full_history_reopens_with_all_snapshots_and_fixed_ids(tmp_path: Path) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    rows = history()
    for index, record in enumerate(rows):
        store.capture(f"source-{index}", record)
        assert (
            store.prepare(record["id"], key=f"prepare-{index}", at=AT)["status"]
            == "ready"
        )
    reopened = journal.Journal(tmp_path, "o/r", 1)
    assert reopened.public_records() == rows
    original = reopened.path.read_bytes()
    reopened.capture("another-delivery", rows[2])
    assert reopened.path.read_bytes() == original
    changed = copy.deepcopy(rows[2])
    changed["data"]["summary"] = "Must not replace the received report"
    with pytest.raises(PublicationError, match="source_changed"):
        reopened.capture("source-2", changed)
    assert reopened.path.read_bytes() == original


WRITER = """
import copy, json, sys
from pathlib import Path
from decision_logs.journal import Journal
store = Journal(Path(sys.argv[1]), "o/r", 1)
raw = json.loads(Path(sys.argv[2]).read_text())
print("ready", flush=True)
assert sys.stdin.readline().strip() == "go"
shared = store.capture("shared", raw)
for i in range(4):
    record = copy.deepcopy(raw)
    record["data"]["summary"] = "writer " + sys.argv[3] + " observation " + str(i)
    store.capture(sys.argv[3] + ":" + str(i), record)
print(shared["id"], flush=True)
"""


def test_two_processes_preserve_all_records_and_share_idempotency_key(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[3]
    template = tmp_path / "input.json"
    raw = result()
    del raw["id"]
    template.write_bytes(schema.json_bytes(raw))
    environment = {**os.environ, "PYTHONPATH": str(root / ".agents/ops")}
    processes = [
        subprocess.Popen(
            [
                str(root / ".venv/bin/python"),
                "-c",
                WRITER,
                str(tmp_path / "state"),
                str(template),
                str(i),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for i in range(2)
    ]
    try:
        for process in processes:
            assert process.stdout is not None
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        outputs = [process.communicate(timeout=15) for process in processes]
        assert all(p.returncode == 0 for p in processes), outputs
        assert outputs[0][0].strip() == outputs[1][0].strip()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
    entries = journal.Journal(tmp_path / "state", "o/r", 1).read()["entries"]
    assert len(entries) == 9
    assert len({e["record"]["id"] for e in entries}) == 9
    assert {e["key"] for e in entries} == {
        "shared",
        *(f"{i}:{j}" for i in range(2) for j in range(4)),
    }


@pytest.mark.parametrize(
    "failed_fsync", [1, 2], ids=["before-replace", "after-replace"]
)
def test_fsync_failure_keeps_history_and_replay_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_fsync: int
) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    store.capture("first", result())
    prefix = store.read()["entries"]
    raw = result()
    del raw["id"]
    real_fsync = os.fsync
    kinds = []

    def interrupted(fd: int) -> None:
        kinds.append(stat.S_IFMT(os.fstat(fd).st_mode))
        if len(kinds) == failed_fsync:
            raise OSError("private storage diagnostic")
        real_fsync(fd)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", interrupted)
        with pytest.raises(PublicationError, match="journal_unavailable") as error:
            store.capture("second", raw)
        assert "private storage diagnostic" not in str(error.value)
    assert kinds[0] == stat.S_IFREG
    if failed_fsync == 2:
        assert kinds[1] == stat.S_IFDIR
    assert store.read()["entries"][:1] == prefix
    restored = store.capture("second", raw)
    assert store.capture("second", raw)["id"] == restored["id"]
    assert len(store.read()["entries"]) == 2
    assert list(store.path.parent.glob(".journal-*")) == []


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "duplicate",
        "unknown_header",
        "version",
        "unknown_entry",
        "unknown_private",
        "projection_tamper",
        "sequence",
    ],
)
def test_corruption_is_never_replaced_with_empty_state(
    tmp_path: Path, fault: str
) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    record = store.capture("first", result())
    store.prepare(record["id"], key="prepare", at=AT)
    row = store.read()
    if fault == "unknown_header":
        row["extra"] = True
    elif fault == "version":
        row["version"] = 2
    elif fault == "unknown_entry":
        row["entries"][0]["extra"] = True
    elif fault == "unknown_private":
        row["entries"][0]["private"]["extra"] = "private"
    elif fault == "projection_tamper":
        row["entries"][1]["payload"]["record"]["data"]["summary"] = (
            "Invented public success"
        )
    elif fault == "sequence":
        row["entries"][1]["sequence"] = 1
    raw = schema.json_bytes(row)
    if fault == "truncated":
        raw = raw[:-8]
    elif fault == "duplicate":
        raw = raw.replace(b'"version": 1', b'"version": 1, "version": 1', 1)
    store.path.write_bytes(raw)
    with pytest.raises(PublicationError, match="journal_invalid"):
        store.capture("new", result())
    assert store.path.read_bytes() == raw


@pytest.mark.parametrize("where", ["journal", "directory"])
def test_private_storage_refuses_symlinks(tmp_path: Path, where: str) -> None:
    store = journal.Journal(tmp_path / "state", "o/r", 1)
    if where == "journal":
        store.path.parent.mkdir(parents=True)
        other = tmp_path / "other.json"
        other.write_text("private original")
        store.path.symlink_to(other)
    else:
        other = tmp_path / "other"
        other.mkdir()
        store.path.parents[2].parent.mkdir(parents=True)
        store.path.parents[2].symlink_to(other, target_is_directory=True)
    with pytest.raises(PublicationError, match="private_storage_symlink"):
        store.capture("first", result())
    if where == "journal":
        assert other.read_text() == "private original"
    else:
        assert list(other.iterdir()) == []


def test_blocked_dependency_retries_only_saved_projection(tmp_path: Path) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    observation = store.capture("result", result())
    judgement = store.capture("decision", decision([observation]))
    blocked = store.prepare(judgement["id"], key="too-early", at=AT)
    assert (
        blocked["status"] == "blocked" and blocked["error"] == "invalid_log_reference"
    )
    assert store.public_records() == []
    store.prepare(observation["id"], key="result-ready", at=AT)
    assert (
        store.prepare(judgement["id"], key="not-a-retry", at=AT)["status"] == "blocked"
    )
    assert (
        store.prepare(judgement["id"], key="explicit-retry", at=AT, retry=True)[
            "status"
        ]
        == "ready"
    )
    assert store.public_records() == [observation, judgement]
    assert len([e for e in store.read()["entries"] if e["event"] == "capture"]) == 2
    assert (
        len(
            [
                e
                for e in store.read()["entries"]
                if e["event"] == "prepare" and e["record_id"] == judgement["id"]
            ]
        )
        == 2
    )


@pytest.mark.parametrize("fault", ["without_preparation", "missing_public_evidence"])
def test_publication_cannot_claim_unverified_success(
    tmp_path: Path, fault: str
) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    record = store.capture("first", result())
    if fault == "missing_public_evidence":
        store.prepare(record["id"], key="ready", at=AT)
    before = store.path.read_bytes()
    with pytest.raises(PublicationError):
        store.publication(
            record["id"], key="fake-success", status="published", at=AT, evidence=[]
        )
    assert store.path.read_bytes() == before


@pytest.mark.parametrize("location", ["summary", "metadata"])
def test_unsafe_projection_is_blocked_and_raw_stays_private(
    tmp_path: Path, location: str
) -> None:
    store = journal.Journal(tmp_path, "o/r", 1)
    raw = result()
    secret = "ghp_" + "x" * 30
    if location == "summary":
        raw["data"]["summary"] = secret
    else:
        raw["data"]["evidence"][0]["url"] = "https://example.com/" + secret
    record = store.capture("unsafe", raw)
    entry = store.prepare(record["id"], key="check", at=AT)
    assert (
        entry["status"] == "blocked"
        and entry["error"] == "credential"
        and entry["payload"] is None
    )
    assert store.public_records() == []
    assert store.read()["entries"][0]["record"] == raw
    assert store.status()[0]["capture_status"] == "captured"
    assert store.status()[0]["publication_status"] == "blocked"
    assert secret not in json.dumps(store.status())
