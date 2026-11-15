"""Actual writes, torn control records and explicit journal repair contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.support.agents_ops.storage_crashes import run_child
from tests.support.agents_ops.storage_tier import OPS, Lab, make_lab


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    monkeypatch.syspath_prepend(str(OPS))
    return make_lab(tmp_path)


def test_trace_detects_unregistered_existing_file_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.agents_ops.storage_crashes import prepare

    monkeypatch.syspath_prepend(str(OPS))
    lab, _, original = prepare("mirror-new", tmp_path)
    registered = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "support/agents_ops/fixtures/storage-crash-boundaries.json"
        ).read_text()
    )["mirror-new"]

    def action() -> object:
        result = original()
        # This existing-file mutation adds no creation, fsync or append_json call.
        fd = os.open(lab.config.state / "lock", os.O_RDWR)
        try:
            os.write(fd, b"unregistered persistent step")
        finally:
            os.close(fd)
        return result

    code, trace = run_child(lab, action, tmp_path / "trace.jsonl")
    assert code == 0 and trace[:-1] == registered
    assert trace[-1] == "write:os:D/.storage-tier/lock"
    with pytest.raises(AssertionError):
        assert trace == registered, "persistence inventory contains an uncovered write"


@pytest.mark.parametrize("role", ["audit", "removals"])
def test_torn_journal_requires_explicit_lossless_repair(lab: Lab, role: str) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    lab.filesystem.write_json(
        lab.config.state / "active.json",
        {
            "schema_version": 1,
            "repository": str(lab.config.repo),
            "mirror_mount": str(lab.config.mount),
        },
    )
    journal = (
        lab.config.state / "audit.jsonl"
        if role == "audit"
        else lab.config.state / "removals" / ("a" * 32 + ".jsonl")
    )
    journal.parent.mkdir(exist_ok=True)
    valid = b'{"event": "known-complete"}\n'
    torn = b'{"event": "uncommitted-tail", "entry":'
    journal.write_bytes(valid + torn)
    before = {
        p: p.read_bytes()
        for p in (lab.config.mirror / unit.path).rglob("*")
        if p.is_file()
    }
    with pytest.raises(lab.filesystem.StorageError, match="journal"):
        lab.engine.run("cycle", [], dry_run=False)
    assert journal.read_bytes() == valid + torn
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, report
    repairs = [r for r in report["results"] if r["action"] == "journal_repaired"]
    assert len(repairs) == 1
    assert Path(repairs[0]["archive"]).read_bytes() == valid + torn
    assert repairs[0]["discarded_tail_bytes"] == len(torn)
    assert journal.read_bytes().startswith(valid)
    assert all(
        isinstance(json.loads(line), dict) for line in journal.read_bytes().splitlines()
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    assert lab.engine.run("recover", [], dry_run=False)[1]["results"] == []
    assert lab.run("verify", unit)[0] == 0
    assert lab.engine.run("cycle", [], dry_run=False)[0] == 0


@pytest.mark.parametrize(
    "api",
    [
        "open-create",
        "open-truncate",
        "open-append",
        "os-write",
        "pwrite",
        "writev",
        "truncate",
        "ftruncate",
        "stream-truncate",
        "builtin-write",
    ],
)
def test_trace_observes_each_write_api(lab: Lab, tmp_path: Path, api: str) -> None:
    path = lab.config.state / "existing.bin"
    path.parent.mkdir(exist_ok=True)
    if api != "open-create":
        path.write_bytes(b"before")

    def action() -> None:
        flags = os.O_RDWR | {
            "open-create": os.O_CREAT,
            "open-truncate": os.O_TRUNC,
            "open-append": os.O_APPEND,
        }.get(api, 0)
        fd = os.open(path, flags)
        try:
            if api == "os-write":
                os.write(fd, b"new")
            elif api == "pwrite":
                os.pwrite(fd, b"new", 0)
            elif api == "writev":
                os.writev(fd, [b"n", b"ew"])
            elif api == "truncate":
                os.truncate(path, 2)
            elif api == "ftruncate":
                os.ftruncate(fd, 2)
            elif api == "stream-truncate":
                with path.open("r+b") as stream:
                    stream.truncate(2)
            elif api == "builtin-write":
                with open(path, "ab") as stream:
                    stream.write(b"new")
        finally:
            os.close(fd)

    code, trace = run_child(lab, action, tmp_path / "trace.jsonl")
    assert code == 0
    assert any(
        value.startswith(("open-os:", "write:", "truncate:")) for value in trace
    ), trace


@pytest.mark.parametrize(
    "role,scenario",
    [
        ("manifests", "mirror-new"),
        ("transactions", "evict"),
        ("publications", "mirror-new"),
        ("checkpoint-file-operations", "checkpoint-mirror"),
    ],
)
def test_partial_atomic_json_preserves_committed_proof_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, scenario: str
) -> None:
    from tests.support.agents_ops.storage_crashes import finish_recovery, prepare

    monkeypatch.syspath_prepend(str(OPS))
    lab, unit, action = prepare(scenario, tmp_path)
    previous = {
        p: p.read_bytes()
        for base in (lab.config.repo, lab.config.mirror)
        for p in (base / ".storage-tier" / role).glob("*.json")
    }
    code, trace = run_child(
        lab,
        action,
        tmp_path / "partial.jsonl",
        tear_when=lambda name: f"/{role}/" in name and name.endswith(".tmp"),
    )
    assert code == 91 and trace[-1].startswith("torn-write:"), trace
    partial = [
        p
        for base in (lab.config.repo, lab.config.mirror)
        for p in (base / ".storage-tier" / role).glob(".*.tmp")
    ]
    assert len(partial) == 1 and partial[0].stat().st_size >= 64
    raw = partial[0].read_bytes()
    with pytest.raises((ValueError, UnicodeError)):
        json.loads(raw)
    assert all(p.read_bytes() == data for p, data in previous.items())
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, report
    assert any(
        r["path"] == str(partial[0]) and r["committed"] is False
        for r in report["uncommitted_json"]
    )
    finish_recovery(lab, unit)
    assert partial[0].read_bytes() == raw


@pytest.mark.parametrize("command", ["mirror", "hydrate"])
def test_multichunk_partial_copy_is_preserved_and_daily_cycle_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from tests.support.agents_ops.storage_crashes import finish_recovery, prepare

    monkeypatch.syspath_prepend(str(OPS))
    lab, unit, action = prepare("mirror-new", tmp_path)
    path = lab.config.repo / unit.path / "a.bin"
    data = b"first-block" * (8 * 1024 * 1024 // 11 + 1) + b"last-block" * 100_000
    path.write_bytes(data)
    lab.wanted["a.bin"] = data
    if command == "hydrate":
        lab.mirror(unit)
        lab.age(unit, 15)
        assert lab.run("evict", unit)[0] == 0

        def action() -> object:
            return lab.run("hydrate", unit)

    code, trace = run_child(
        lab,
        action,
        tmp_path / "partial-copy.jsonl",
        stop_when=lambda name: (
            name.startswith("write:stream-1:") and name.endswith("/a.bin")
        ),
    )
    assert code == 91, trace
    base = lab.config.mirror if command == "mirror" else lab.config.repo
    copies = list((base / unit.path).parent.glob(".storage-tier-*/a.bin"))
    assert len(copies) == 1
    assert copies[0].read_bytes() == data[: 8 * 1024 * 1024]
    inode = copies[0].stat().st_dev, copies[0].stat().st_ino
    finish_recovery(lab, unit)
    archives = list((base / ".storage-tier/recovered-publications").rglob("a.bin"))
    assert any(
        (p.stat().st_dev, p.stat().st_ino) == inode
        and p.read_bytes() == data[: 8 * 1024 * 1024]
        for p in archives
    )


@pytest.mark.parametrize("role", ["audit", "removals"])
def test_process_exit_inside_append_preserves_tail_before_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    from tests.support.agents_ops.storage_crashes import finish_recovery, prepare

    monkeypatch.syspath_prepend(str(OPS))
    lab, unit, action = prepare("evict", tmp_path)
    code, trace = run_child(
        lab,
        action,
        tmp_path / "torn-append.jsonl",
        tear_when=lambda name: (
            name.endswith("audit.jsonl")
            if role == "audit"
            else "/removals/" in name and name.endswith(".jsonl")
        ),
    )
    assert code == 91 and trace[-1].startswith("torn-write:"), trace
    folder = lab.config.state / "removals"
    paths = (
        [lab.config.state / "audit.jsonl"]
        if role == "audit"
        else list(folder.glob("*.jsonl"))
    )
    torn = {
        p: p.read_bytes()
        for p in paths
        if p.read_bytes() and not p.read_bytes().endswith(b"\n")
    }
    assert len(torn) == 1
    with pytest.raises(lab.filesystem.StorageError, match="journal"):
        lab.run("verify", unit)
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, report
    repair = next(r for r in report["results"] if r["action"] == "journal_repaired")
    assert Path(repair["archive"]).read_bytes() == next(iter(torn.values()))
    finish_recovery(lab, unit)


@pytest.mark.parametrize(
    "role", ["manifests", "transactions", "publications", "checkpoint-file-operations"]
)
def test_corrupt_committed_json_cannot_authorize_recovery(lab: Lab, role: str) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    if role == "manifests":
        path = lab.engine.manifest_path(unit)

        def action():
            return lab.run("verify", unit)
    else:
        path = lab.config.state / role / ("b" * 32 + ".json")

        def action():
            return lab.engine.run("recover", [], dry_run=False)

    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b'{"phase": "prepared", "payload":')
    before = (lab.config.mirror / unit.path / "a.bin").read_bytes()
    with pytest.raises((ValueError, lab.filesystem.StorageError)):
        action()
    assert path.read_bytes() == b'{"phase": "prepared", "payload":'
    assert (lab.config.mirror / unit.path / "a.bin").read_bytes() == before


def test_interior_journal_damage_requires_manual_inspection(lab: Lab) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = lab.config.state / "audit.jsonl"
    damaged = b'{"event":"valid"}\n{"event":\n{"event":"valid-later"}\n'
    path.write_bytes(damaged)
    with pytest.raises(lab.filesystem.StorageError, match="interior"):
        lab.engine.run("recover", [], dry_run=False)
    assert path.read_bytes() == damaged


def test_large_json_fragment_reaches_disk_without_replacing_saved_evidence(
    lab: Lab, tmp_path: Path
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = lab.engine.manifest_path(unit)
    original = path.read_bytes()
    value = lab.filesystem.read_json(path)
    value["uncommitted_extension"] = "large JSON fragment " * 4096
    code, trace = run_child(
        lab,
        lambda: lab.filesystem.write_json(path, value),
        tmp_path / "large-json.jsonl",
        tear_when=lambda name: "/manifests/" in name and name.endswith(".tmp"),
        tear_after=16 * 1024,
    )
    assert code == 91 and trace[-1].startswith("torn-write:"), trace
    temporary = next(path.parent.glob(".*.tmp"))
    assert temporary.stat().st_size > 8192
    with pytest.raises(ValueError):
        json.loads(temporary.read_bytes())
    assert path.read_bytes() == original
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0 and report["uncommitted_json"]
    assert lab.run("verify", unit)[0] == 0
    assert temporary.exists()


@pytest.mark.parametrize("point", ["intent", "archive"])
def test_journal_repair_refuses_changed_evidence(
    lab: Lab, tmp_path: Path, point: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = lab.config.state / "audit.jsonl"
    torn = b'{"event":"complete"}\n{"event":'
    path.write_bytes(torn)
    code, trace = run_child(
        lab,
        lambda: lab.engine.run("recover", [], dry_run=False),
        tmp_path / "repair-crash.jsonl",
        stop_when=lambda label: (
            label == "json:journal-repairs:prepared"
            if point == "intent"
            else label.startswith("replace:D/.storage-tier/journal-history/")
        ),
    )
    assert code == 91, trace
    changed = (
        path
        if point == "intent"
        else next((lab.config.state / "journal-history").glob("*.jsonl"))
    )
    changed.write_bytes(b"unverified replacement evidence")
    with pytest.raises(lab.filesystem.StorageError, match="evidence changed"):
        lab.engine.run("recover", [], dry_run=False)
    assert changed.read_bytes() == b"unverified replacement evidence"
    assert list((lab.config.state / "journal-repairs").glob("*.json"))


@pytest.mark.parametrize("role,event", [("audit", "deleted"), ("removals", "removed")])
def test_torn_completion_append_recovers_after_payload_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, event: str
) -> None:
    from tests.support.agents_ops.storage_crashes import finish_recovery, prepare

    monkeypatch.syspath_prepend(str(OPS))
    lab, unit, original = prepare("evict", tmp_path)
    append = lab.module.append_json

    def action() -> object:
        def interrupted(path: Path, value: dict) -> None:
            matches = (
                path.name == "audit.jsonl"
                if role == "audit"
                else path.parent.name == "removals"
            )
            if matches and value["event"] == event:
                raw = (json.dumps(value, sort_keys=True) + "\n").encode()
                fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                os.write(fd, raw[: len(raw) // 2])
                os.fsync(fd)
                os._exit(91)
            append(path, value)

        lab.module.append_json = interrupted
        return original()

    code, trace = run_child(lab, action, tmp_path / "late-torn.jsonl")
    assert code == 91, trace
    assert not (lab.config.repo / unit.path / "a.bin").exists()
    journals = (
        [lab.config.state / "audit.jsonl"]
        if role == "audit"
        else list((lab.config.state / "removals").glob("*.jsonl"))
    )
    data = next(
        path.read_bytes() for path in journals if not path.read_bytes().endswith(b"\n")
    )
    code, report = lab.engine.run("recover", [], dry_run=False)
    assert code == 0, report
    repair = next(r for r in report["results"] if r["action"] == "journal_repaired")
    assert Path(repair["archive"]).read_bytes() == data
    finish_recovery(lab, unit)


@pytest.mark.parametrize("kind", ["audit", "manifest"])
def test_control_fifo_is_rejected_before_reading(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    unit = lab.unit()
    lab.mirror(unit)
    path = (
        lab.config.state / "audit.jsonl"
        if kind == "audit"
        else lab.engine.manifest_path(unit)
    )
    path.unlink()
    os.mkfifo(path)
    owner = lab.module.journals if kind == "audit" else lab.filesystem
    read = owner.read_bytes

    def checked(candidate: Path) -> bytes:
        assert candidate != path, "special control file must be rejected before open"
        data = read(candidate)
        assert isinstance(data, bytes)
        return data

    monkeypatch.setattr(owner, "read_bytes", checked)
    with pytest.raises(lab.filesystem.StorageError, match="journal|control"):
        lab.engine.run("recover", [], dry_run=False)
