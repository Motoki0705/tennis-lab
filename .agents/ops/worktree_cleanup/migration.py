"""Reconcile device-based records against pinned Git registrations once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import inventory
from .storage import CleanupError, State, utc

TERMINAL = {"deleted", "absent", "superseded"}
UNFINISHED = {"stopping", "quarantined", "unregistered", "quarantine_conflict"}


def residual_quarantine(saved: dict[str, Any], repo: Path) -> Path | None:
    quarantine = repo / ".claude/worktrees" / f".expired-{saved['id']}"
    if saved.get("quarantine") not in {None, str(quarantine)}:
        raise CleanupError("legacy intent has an invalid quarantine path")
    try:
        quarantine.lstat()
    except FileNotFoundError:
        return None
    return quarantine


def require_ready(saved: dict[str, Any]) -> None:
    event = saved.get("filesystem_migration", {})
    if event.get("result") == "blocked":
        raise CleanupError(event["reason"])


def directory_evidence(
    saved: dict[str, Any], current: dict[str, Any], repo: Path
) -> tuple[dict[str, Any] | None, str]:
    root = Path(saved["path"])
    physical = root
    if saved.get("quarantine"):
        quarantine = Path(saved["quarantine"])
        if quarantine != root.parent / (".expired-" + saved["id"]):
            return None, "legacy quarantine path does not match the intent"
        if os.path.lexists(quarantine):
            physical = quarantine
    if inventory.scope_error(repo, physical):
        return None, "legacy worktree path was replaced or redirected"
    expected = saved.get("directory_identity")
    if not isinstance(expected, dict):
        return None, "legacy worktree has no recorded directory identity"
    try:
        admin = inventory.open_directory(Path(current["admin"]))
        try:
            if inventory.admin_identity(admin) != current["admin_identity"]:
                return None, "Git administrative generation changed during migration"
            backlink = inventory.read_at(admin, "gitdir").rstrip("\n")
            if (
                Path(os.path.abspath(Path(current["admin"]) / backlink))
                != root / ".git"
            ):
                return None, "Git administrative backlink differs from the saved path"
            leaf = inventory.open_directory(physical)
            try:
                actual = inventory.opened_identity(leaf)
                if actual["birth_time_ns"] is None:
                    return None, "worktree directory birth time is unavailable"
                if (
                    expected["inode"] != actual["inode"]
                    or (
                        expected.get("filesystem_id") is not None
                        and expected["filesystem_id"] != actual["filesystem_id"]
                    )
                    or (
                        expected.get("birth_time_ns") is not None
                        and expected["birth_time_ns"] != actual["birth_time_ns"]
                    )
                ):
                    return None, "legacy worktree directory generation differs"
                link = inventory.read_at(leaf, ".git").strip()
                if not link.startswith("gitdir: ") or Path(
                    os.path.abspath(physical / link.removeprefix("gitdir: "))
                ) != Path(current["admin"]):
                    return None, "worktree Git file refers to another registration"
                if inventory.identity(physical) != actual:
                    return None, "worktree directory changed during migration"
            finally:
                os.close(leaf)
            if inventory.admin_identity(admin) != current["admin_identity"]:
                return None, "Git administrative generation changed during migration"
            return (
                actual,
                "inode, administrative birth time and bidirectional Git registration match",
            )
        finally:
            os.close(admin)
    except (FileNotFoundError, NotADirectoryError):
        return None, "legacy worktree or Git registration disappeared"


def registration_evidence(
    saved: dict[str, Any], current: dict[str, Any] | None, repo: Path
) -> tuple[dict[str, Any] | None, str]:
    if current is None:
        return None, "legacy Git registration disappeared"
    old, ident = saved["admin_identity"], current["admin_identity"]
    if saved["path"] != current["path"]:
        return None, "registered worktree path differs from the legacy path"
    if (
        old["inode"] != ident["inode"]
        or old.get("birth_time_ns") is None
        or old["birth_time_ns"] != ident["birth_time_ns"]
        or (old.get("token") is not None and old["token"] != ident.get("token"))
    ):
        return (
            None,
            "administrative inode, birth time or generation token differs or is unavailable",
        )
    if current["scope_error"]:
        return None, "current worktree registration is outside the safe scope"
    return directory_evidence(saved, current, repo)


def legacy_filesystems(
    state: State, registrations: dict[str, dict[str, Any]]
) -> dict[int, set[str]]:
    """Map old device numbers using currently verified, independent generations."""
    filesystems: dict[int, set[str]] = {}
    for saved in state.records.values():
        candidates = [saved]
        previous = saved.get("filesystem_migration", {})
        if isinstance(previous.get("previous_admin_identity"), dict):
            candidates.append(
                {
                    **saved,
                    "admin_identity": previous["previous_admin_identity"],
                    "directory_identity": previous.get("previous_directory_identity"),
                }
            )
        for candidate in candidates:
            old = candidate["admin_identity"]
            current = registrations.get(candidate["admin"])
            if "device" not in old or current is None:
                continue
            try:
                actual, _ = registration_evidence(
                    candidate, current, Path(state.data["repo"])
                )
            except (OSError, ValueError, CleanupError):
                continue
            if actual is None:
                continue
            filesystems.setdefault(old["device"], set()).add(
                current["admin_identity"]["filesystem_id"]
            )
            directory = candidate["directory_identity"]
            if "device" in directory:
                filesystems.setdefault(directory["device"], set()).add(
                    actual["filesystem_id"]
                )
    return filesystems


def quarantine_evidence(
    saved: dict[str, Any], repo: Path, filesystems: dict[int, set[str]]
) -> tuple[dict[str, Any] | None, str]:
    quarantine = repo / ".claude/worktrees" / f".expired-{saved['id']}"
    if error := inventory.scope_error(repo, quarantine):
        return None, error
    expected = saved.get("directory_identity")
    if not isinstance(expected, dict):
        return None, "legacy quarantine has no recorded directory identity"
    leaf = inventory.open_directory(quarantine)
    try:
        actual = inventory.opened_identity(leaf)
        filesystem = expected.get("filesystem_id")
        if filesystem is None:
            mapped = filesystems.get(expected["device"], set())
            if len(mapped) != 1:
                return (
                    None,
                    "legacy quarantine filesystem has no unambiguous pinned Git witness",
                )
            filesystem = next(iter(mapped))
        # The oldest format saved only the root inode. An exact match to the
        # independently saved admin btime can identify a co-created root too.
        born = expected.get("birth_time_ns")
        if born is None:
            born = saved["admin_identity"].get("birth_time_ns")
        if born is None or (
            actual
            != {
                "filesystem_id": filesystem,
                "inode": expected["inode"],
                "birth_time_ns": born,
            }
        ):
            return (
                None,
                "legacy quarantine filesystem, inode or birth time differs or is unavailable",
            )
        link = inventory.read_at(leaf, ".git").strip()
        if not link.startswith("gitdir: ") or Path(
            os.path.abspath(quarantine / link.removeprefix("gitdir: "))
        ) != Path(saved["admin"]):
            return None, "worktree Git file refers to another registration"
        if inventory.identity(quarantine) != actual:
            return None, "legacy quarantine changed during migration"
        return (
            actual,
            "quarantine path, filesystem, inode, exact saved birth time and Git file match. Registration authority revoked",
        )
    finally:
        os.close(leaf)


def reconcile(state: State, rows: list[dict[str, Any]], now: int) -> None:
    """Publish a complete migration through State's atomic, fsynced save."""
    registrations = {row["admin"]: row for row in rows}
    filesystems = legacy_filesystems(state, registrations)
    repo = Path(state.data["repo"])
    changed = state.data["schema_version"] != 2
    for saved in state.records.values():
        event = saved.get("filesystem_migration", {})
        recoverable = (
            saved["phase"] == "superseded"
            and event.get("result") == "retired"
            and event.get("previous_phase") in UNFINISHED
            and event.get("previous_admin_identity") == saved["admin_identity"]
            and event.get("previous_directory_identity")
            == saved.get("directory_identity")
            and saved.get("superseded_at") == event.get("checked_at")
        )
        if (
            "device" not in saved["admin_identity"]
            or (saved["phase"] in TERMINAL and not recoverable)
            or (event.get("result") == "migrated" and event.get("registration_lost"))
        ):
            continue
        residue = False
        inspection_error = None
        if saved["phase"] in UNFINISHED or recoverable:
            try:
                quarantine = residual_quarantine(saved, repo)
                residue = quarantine is not None
                if quarantine is not None:
                    saved["quarantine"] = str(quarantine)
            except (OSError, ValueError, CleanupError) as error:
                residue, inspection_error = True, str(error)
        if recoverable:
            if not residue:
                continue
            saved["phase"] = event["previous_phase"]
            saved.pop("superseded_at", None)
        old = saved["admin_identity"]
        changed = True
        current = registrations.get(saved["admin"])
        try:
            actual, reason = registration_evidence(saved, current, repo)
        except (OSError, ValueError, CleanupError) as error:
            actual, reason = None, str(error)
        outcome = "migrated" if actual is not None else "retired"
        registration_lost = False
        if actual is None and residue:
            detail = inspection_error
            if detail is None:
                try:
                    actual, detail = quarantine_evidence(saved, repo, filesystems)
                except (OSError, ValueError, CleanupError) as error:
                    detail = str(error)
            if actual is not None:
                outcome, reason, registration_lost = "migrated", str(detail), True
                saved.setdefault("registration_lost_at", utc(now))
            else:
                outcome, reason = "blocked", reason + ". " + str(detail)
        retirement = event if recoverable else event.get("recovered_retirement")
        saved["filesystem_migration"] = {
            "result": outcome,
            "checked_at": utc(now),
            "reason": reason,
            "previous_phase": saved["phase"],
            "previous_admin_identity": old,
            "previous_directory_identity": saved.get("directory_identity"),
            **({"registration_lost": True} if registration_lost else {}),
            **({"recovered_retirement": retirement} if retirement else {}),
        }
        if actual is not None:
            saved["directory_identity"] = actual
            if not registration_lost and current is not None:
                saved["admin_identity"] = current["admin_identity"]
                saved.pop("registration_lost_at", None)
        elif outcome != "blocked":
            saved.update(phase="superseded", superseded_at=utc(now))
    state.data["schema_version"] = 2
    if changed:
        state.save()
