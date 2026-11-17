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


def detached_legacy_record(
    lab: Lab, phase: str = "quarantined", *, retired: bool = False
) -> dict[str, Any]:
    data = legacy_records(lab, 2)
    old = next(row for row in data["records"].values() if row["phase"] == "stopping")
    Path(old["path"]).rename(old["quarantine"])
    lab.git("worktree", "remove", "--force", old["path"])
    old["phase"] = phase
    if retired:
        old.update(phase="superseded", superseded_at=lab.storage.utc(lab.now))
        old["filesystem_migration"] = {
            "result": "retired",
            "checked_at": old["superseded_at"],
            "reason": "legacy Git registration disappeared",
            "previous_phase": phase,
            "previous_admin_identity": dict(old["admin_identity"]),
            "previous_directory_identity": dict(old["directory_identity"]),
        }
        data["schema_version"] = 2
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    return dict(old)


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


def test_quarantined_git_file_mismatch_keeps_the_residue_as_a_failure(lab: Lab) -> None:
    data = legacy_records(lab, 2)
    row = next(row for row in data["records"].values() if row["phase"] == "quarantined")
    other = next(row for row in data["records"].values() if row["phase"] == "stopping")
    link = Path(row["quarantine"]) / ".git"
    changed = "gitdir: " + other["admin"] + "\n"
    link.write_text(changed)
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True, dry_run=True)
    assert code == 1 and report["outcome"] == "partial_failure"
    saved = lab.records()[row["id"]]
    assert saved["phase"] == "quarantined"
    assert saved["filesystem_migration"]["result"] == "blocked"
    assert (
        "Git file refers to another registration"
        in saved["filesystem_migration"]["reason"]
    )
    assert link.read_text() == changed
    assert row["id"] in {entry["id"] for entry in report["entries"]}


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize(
    "phase", ["stopping", "quarantined", "unregistered", "quarantine_conflict"]
)
def test_unverifiable_legacy_intent_keeps_residual_quarantine_as_a_failure(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, replacement: bool, phase: str
) -> None:
    data = legacy_records(lab)
    old = next(iter(data["records"].values()))
    path, quarantine = Path(old["path"]), Path(old["quarantine"])
    path.rename(quarantine)
    lab.git("worktree", "remove", "--force", str(path))
    old["phase"] = phase
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
        assert current["phase"] == phase
        assert current["filesystem_migration"]["result"] == "blocked"
    code, report = lab.cleaner().run(cleanup=True, dry_run=True)
    assert code == 1 and report["errors"]
    assert quarantine.is_dir()
    shutil.rmtree(quarantine)
    lab.run(cleanup=True)
    assert lab.records()[old["id"]]["phase"] == "superseded"
    if replacement:
        assert (path / "keep").read_text() == "replacement"


@pytest.mark.parametrize("retired", [False, True])
@pytest.mark.parametrize(
    "phase", ["stopping", "quarantined", "unregistered", "quarantine_conflict"]
)
def test_verified_detached_legacy_residue_migrates_once_and_is_deleted(
    lab: Lab, phase: str, retired: bool
) -> None:
    old = detached_legacy_record(lab, phase, retired=retired)
    quarantine = Path(old["quarantine"])
    lab.now += LIFETIME
    first = lab.run(cleanup=True, dry_run=True)
    entry = next(entry for entry in first["entries"] if entry["id"] == old["id"])
    assert entry["selected"] is True and entry["quarantine"] == str(quarantine)
    check = next(row for row in first["stop_preflight"] if row["path"] == old["path"])
    assert check["reference_gate"]["stage"] == "after_quarantine"
    saved = lab.records()[old["id"]]
    assert saved["created_at_ns"] == BORN
    assert saved["phase"] == phase
    assert saved["directory_identity"] == lab.inventory.identity(quarantine)
    assert saved["filesystem_migration"]["result"] == "migrated"
    if retired:
        assert (
            saved["filesystem_migration"]["recovered_retirement"]
            == old["filesystem_migration"]
        )
    second = lab.run(cleanup=True, dry_run=True)
    assert second["filesystem_migrations"] == first["filesystem_migrations"]
    assert lab.records()[old["id"]] == saved
    report = lab.run(cleanup=True)
    result = next(row for row in report["results"] if row["path"] == old["path"])
    assert result["deleted"] is True
    assert not quarantine.exists()
    assert lab.records()[old["id"]]["phase"] == "deleted"
    journal = [
        json.loads(line)
        for line in (lab.state / "results.jsonl").read_text().splitlines()
    ]
    assert any(row["id"] == old["id"] and row["deleted"] for row in journal)
    assert lab.run(cleanup=True)["results"] == []


def test_detached_residue_with_saved_filesystem_and_birth_needs_no_git_witness(
    lab: Lab,
) -> None:
    data = legacy_records(lab)
    old = next(iter(data["records"].values()))
    path, quarantine = Path(old["path"]), Path(old["quarantine"])
    old["directory_identity"] = lab.inventory.identity(path)
    path.rename(quarantine)
    lab.git("worktree", "remove", "--force", str(path))
    old["phase"] = "quarantined"
    old.pop("quarantine")
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    lab.now += LIFETIME
    preview = lab.run(cleanup=True, dry_run=True)
    assert preview["entries"][0]["quarantine"] == str(quarantine)
    assert preview["stop_preflight"][0]["reference_gate"]["stage"] == "after_quarantine"
    report = lab.run(cleanup=True)
    assert report["results"][0]["deleted"] is True
    assert not quarantine.exists()


@pytest.mark.parametrize(
    "change",
    [
        "inode",
        "birth_time",
        "filesystem_id",
        "git_file",
        "admin_birth",
        "witness",
        "symlink",
    ],
)
def test_detached_residue_with_insufficient_identity_stays_visible_and_fails(
    lab: Lab, change: str
) -> None:
    old = detached_legacy_record(lab, retired=True)
    data = json.loads((lab.state / "worktrees.json").read_text())
    row = data["records"][old["id"]]
    quarantine = Path(old["quarantine"])
    if change == "inode":
        row["directory_identity"]["inode"] += 1
    elif change == "birth_time":
        row["directory_identity"]["birth_time_ns"] = BORN - 1
    elif change == "filesystem_id":
        row["directory_identity"] = {
            **lab.inventory.identity(quarantine),
            "filesystem_id": "ext4-fsid:0000000000000001",
        }
    elif change == "git_file":
        (quarantine / ".git").write_text("gitdir: /another/registration\n")
    elif change == "admin_birth":
        row["admin_identity"]["birth_time_ns"] = None
    elif change == "witness":
        anchor = next(r for r in data["records"].values() if r["id"] != old["id"])
        anchor["admin_identity"]["birth_time_ns"] -= 1
    else:
        moved = quarantine.with_name("keep-directory")
        quarantine.rename(moved)
        quarantine.symlink_to(moved, target_is_directory=True)
    # The old retirement captured the same unprovable evidence.
    row["filesystem_migration"]["previous_admin_identity"] = dict(row["admin_identity"])
    row["filesystem_migration"]["previous_directory_identity"] = dict(
        row["directory_identity"]
    )
    (lab.state / "worktrees.json").write_text(json.dumps(data))
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True, dry_run=True)
    assert code == 1 and report["errors"]
    migration = next(r for r in report["filesystem_migrations"] if r["id"] == old["id"])
    assert migration["result"] == "blocked" and migration["quarantine"] == str(
        quarantine
    )
    assert lab.records()[old["id"]]["phase"] == "quarantined"
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1 and report["outcome"] == "partial_failure"
    result = next(r for r in report["results"] if r["path"] == old["path"])
    assert result["deleted"] is False and result["error"]
    assert quarantine.is_dir()
    assert json.loads((lab.state / "report-latest.json").read_text()) == report
    journal = [
        json.loads(line)
        for line in (lab.state / "results.jsonl").read_text().splitlines()
    ]
    assert any(r["id"] == old["id"] and r["phase"] == "failed" for r in journal)


@pytest.mark.parametrize("failure", ["stop", "remove"])
def test_detached_residue_retries_failures_and_preserves_a_reused_registration(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    old = detached_legacy_record(lab, retired=True)
    path, quarantine = Path(old["path"]), Path(old["quarantine"])
    lab.now += LIFETIME
    monkeypatch.setattr(
        lab.inventory,
        "birth_time",
        lambda fd: (
            lab.now
            if os.readlink(f"/proc/self/fd/{fd}") in {str(path), old["admin"]}
            else BORN
        ),
    )
    lab.add("replacement", path=path)
    (path / "keep").write_text("successor")
    lab.run(cleanup=True, dry_run=True)
    admin = Path(old["admin"])
    token = (admin / lab.inventory.TOKEN_FILE).read_bytes()
    with monkeypatch.context() as patch:
        if failure == "stop":
            real_stop = lab.lifecycle.Stopper.stop_moved

            def stop(stopper: Any, physical: Path, original: Path, stamp: str) -> None:
                if physical == quarantine:
                    raise lab.storage.CleanupError("injected stop failure")
                real_stop(stopper, physical, original, stamp)

            patch.setattr(lab.lifecycle.Stopper, "stop_moved", stop)
        else:
            real_remove = lab.inventory.remove_tree

            def remove(physical: Path, expected: Any) -> None:
                if physical == quarantine:
                    raise OSError("injected removal failure")
                real_remove(physical, expected)

            patch.setattr(lab.inventory, "remove_tree", remove)
        code, report = lab.cleaner().run(cleanup=True)
        assert code == 1
        result = next(r for r in report["results"] if r["path"] == old["path"])
        assert result["deleted"] is False and "injected" in result["error"]
        assert quarantine.exists()
    lab.run(cleanup=True)
    assert not quarantine.exists()
    assert (path / "keep").read_text() == "successor"
    assert (admin / lab.inventory.TOKEN_FILE).read_bytes() == token
    assert str(path) in lab.git("worktree", "list", "--porcelain")
