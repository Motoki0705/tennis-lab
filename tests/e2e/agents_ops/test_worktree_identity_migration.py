"""Legacy expiry records migrate only with independently pinned Git evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_worktree_cleanup import BORN, LIFETIME, Lab
from tests.e2e.agents_ops.test_worktree_cleanup import lab as lab


def legacy_records(lab: Lab, count: int = 1) -> dict[str, Any]:
    for index in range(count):
        lab.add("task-" + str(index))
    lab.run()
    data = json.loads((lab.state / "worktrees.json").read_text())
    rows = {}
    for index, row in enumerate(data["records"].values()):
        for key in ("admin_identity", "directory_identity"):
            ident = row[key]
            ident.pop("filesystem_id")
            ident["device"] = 999999
        row["directory_identity"].pop("birth_time_ns")
        ident = row["admin_identity"]
        row["id"] = hashlib.sha256(
            f"{row['admin']}:{ident['device']}:{ident['inode']}:{ident['birth_time_ns']}:{ident['token']}".encode()
        ).hexdigest()
        row["phase"] = "stopping" if index % 2 == 0 else "quarantined"
        row["quarantine"] = str(Path(row["path"]).parent / (".expired-" + row["id"]))
        if row["phase"] == "quarantined":
            Path(row["path"]).rename(row["quarantine"])
        rows[row["id"]] = row
    data.update(schema_version=1, records=rows)
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    return dict(data)


def test_six_legacy_intents_migrate_once_and_resume_across_device_changes(
    lab: Lab,
) -> None:
    old = legacy_records(lab, 6)
    lab.now += LIFETIME
    first = lab.run(cleanup=True, dry_run=True)
    assert len(first["entries"]) == len(first["stop_preflight"]) == 6
    assert len({row["path"] for row in first["entries"]}) == 6
    assert {row["id"] for row in first["entries"]} == set(old["records"])
    assert {row["result"] for row in first["filesystem_migrations"]} == {"migrated"}
    records = lab.records()
    for key, row in records.items():
        assert row["created_at_ns"] == BORN
        assert row["quarantine"] == old["records"][key]["quarantine"]
        assert "filesystem_id" in row["admin_identity"]
        assert "filesystem_id" in row["directory_identity"]
        assert row["directory_identity"]["birth_time_ns"] == BORN
    second = lab.run(cleanup=True, dry_run=True)
    assert second["filesystem_migrations"] == first["filesystem_migrations"]
    assert lab.records() == records
    report = lab.run(cleanup=True)
    assert len(report["results"]) == 6 and all(
        row["deleted"] for row in report["results"]
    )
    assert not list((lab.repo / ".claude/worktrees").glob(".expired-*"))


@pytest.mark.parametrize(
    "change",
    [
        "admin_birth",
        "root_birth",
        "admin_inode",
        "root_inode",
        "path",
        "registration",
        "unknown_birth",
    ],
)
def test_mismatching_legacy_intent_retires_and_current_registration_uses_birth_time(
    lab: Lab, change: str
) -> None:
    data = legacy_records(lab)
    old = next(iter(data["records"].values()))
    if change == "admin_birth":
        old["admin_identity"]["birth_time_ns"] -= 1
    elif change == "root_birth":
        old["directory_identity"]["birth_time_ns"] = BORN - 1
    elif change == "admin_inode":
        old["admin_identity"]["inode"] += 1
    elif change == "root_inode":
        old["directory_identity"]["inode"] += 1
    elif change == "path":
        old["path"] += "-previous"
    elif change == "registration":
        old["admin"] += "-previous"
    else:
        old["admin_identity"]["birth_time_ns"] = None
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    lab.now += LIFETIME
    report = lab.run(cleanup=True, dry_run=True)
    assert len(report["entries"]) == 1
    assert report["entries"][0]["id"] != old["id"]
    assert report["entries"][0]["created_at"] == lab.storage.utc(BORN)
    retired = lab.records()[old["id"]]
    assert retired["phase"] == "superseded"
    assert retired["filesystem_migration"]["result"] == "retired"
    assert retired["filesystem_migration"]["reason"]
    lab.run(cleanup=True, dry_run=True)
    assert len(lab.records()) == 2


@pytest.mark.parametrize("published", [False, True])
def test_migration_retries_after_atomic_state_publication_is_interrupted(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    data = legacy_records(lab)
    original = lab.storage.write_json

    class Interrupted(BaseException):
        pass

    with monkeypatch.context() as patch:

        def write(path: Path, value: Any) -> None:
            if path == lab.state / "worktrees.json" and value["schema_version"] == 2:
                if published:
                    original(path, value)
                raise Interrupted()
            original(path, value)

        patch.setattr(lab.storage, "write_json", write)
        with pytest.raises(Interrupted):
            lab.run()
    report = lab.run()
    assert len(report["entries"]) == 1
    assert report["entries"][0]["id"] in data["records"]
    assert len(lab.records()) == 1
    assert report["filesystem_migrations"][0]["result"] == "migrated"


def test_disappeared_legacy_intent_is_closed_without_a_permanent_failure(
    lab: Lab,
) -> None:
    old = next(iter(legacy_records(lab)["records"].values()))
    lab.git("worktree", "remove", old["path"])
    report = lab.run(cleanup=True)
    assert report["entries"] == [] and report["results"] == []
    assert lab.records()[old["id"]]["phase"] == "superseded"
    assert report["filesystem_migrations"][0]["result"] == "retired"


def test_unavailable_filesystem_identity_fails_before_state_or_deletion(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()

    def unavailable(fd: int) -> str:
        raise lab.storage.CleanupError("persistent filesystem identity unavailable")

    monkeypatch.setattr(lab.inventory, "filesystem_id", unavailable)
    with pytest.raises(
        lab.storage.CleanupError, match="filesystem identity unavailable"
    ):
        lab.run(cleanup=True)
    assert path.is_dir() and not (lab.state / "worktrees.json").exists()


def test_quarantined_git_file_mismatch_retires_the_old_intent(lab: Lab) -> None:
    data = legacy_records(lab, 2)
    row = next(row for row in data["records"].values() if row["phase"] == "quarantined")
    other = next(row for row in data["records"].values() if row["phase"] == "stopping")
    link = Path(row["quarantine"]) / ".git"
    changed = "gitdir: " + other["admin"] + "\n"
    link.write_text(changed)
    report = lab.run()
    retired = lab.records()[row["id"]]
    assert retired["phase"] == "superseded"
    assert (
        retired["filesystem_migration"]["reason"]
        == "worktree Git file refers to another registration"
    )
    assert link.read_text() == changed
    assert row["id"] not in {entry["id"] for entry in report["entries"]}


@pytest.mark.parametrize("replacement", [False, True])
def test_legacy_unregistered_intent_keeps_residual_quarantine_as_a_failure(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, replacement: bool
) -> None:
    data = legacy_records(lab)
    old = next(iter(data["records"].values()))
    path, quarantine = Path(old["path"]), Path(old["quarantine"])
    path.rename(quarantine)
    lab.git("worktree", "remove", "--force", str(path))
    old["phase"] = "unregistered"
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    lab.now += LIFETIME
    if replacement:
        monkeypatch.setattr(
            lab.inventory,
            "birth_time",
            lambda fd: (
                BORN if ".expired-" in os.readlink(f"/proc/self/fd/{fd}") else lab.now
            ),
        )
        lab.add("replacement", path=path)
        (path / "keep").write_text("replacement")
    for _ in range(2):
        code, report = lab.cleaner().run(cleanup=True)
        assert code == 1
        assert report["results"][0]["deleted"] is False
        assert quarantine.is_dir()
        current = lab.records()[old["id"]]
        assert current["phase"] == "unregistered"
        assert current["filesystem_migration"]["result"] == "blocked"
    code, report = lab.cleaner().run(cleanup=True, dry_run=True)
    assert code == 1 and report["errors"]
    assert quarantine.is_dir()
    shutil.rmtree(quarantine)
    lab.run(cleanup=True)
    assert lab.records()[old["id"]]["phase"] == "superseded"
    if replacement:
        assert (path / "keep").read_text() == "replacement"
