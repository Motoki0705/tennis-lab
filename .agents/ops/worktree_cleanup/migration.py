"""Reconcile device-based records against pinned Git registrations once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from . import inventory
from .storage import CleanupError, State, utc

TERMINAL = {"deleted", "absent", "superseded"}


def unregistered_residue(saved: dict[str, Any], repo: Path) -> str | None:
    """Do not retire the filesystem duty left after legacy Git unregistration."""
    if saved["phase"] != "unregistered":
        return None
    quarantine = repo / ".claude/worktrees" / (".expired-" + saved["id"])
    if saved.get("quarantine") not in {None, str(quarantine)}:
        return "legacy unregistered intent has an invalid quarantine path"
    try:
        quarantine.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        return f"legacy unregistered quarantine cannot be inspected: {error}"
    return (
        "legacy unregistered quarantine remains; its generation cannot be verified "
        "without the matching Git registration"
    )


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
                if expected["inode"] != actual["inode"] or (
                    expected.get("birth_time_ns") is not None
                    and expected["birth_time_ns"] != actual["birth_time_ns"]
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


def reconcile(state: State, rows: list[dict[str, Any]], now: int) -> None:
    """Publish a complete migration through State's atomic, fsynced save."""
    registrations = {row["admin"]: row for row in rows}
    changed = state.data["schema_version"] != 2
    for saved in state.records.values():
        if "device" not in saved["admin_identity"] or saved["phase"] in TERMINAL:
            continue
        old = saved["admin_identity"]
        changed = True
        current = registrations.get(saved["admin"])
        reason = "legacy Git registration disappeared"
        actual = None
        if current is not None:
            ident = current["admin_identity"]
            if saved["path"] != current["path"]:
                reason = "registered worktree path differs from the legacy path"
            elif (
                old["inode"] != ident["inode"]
                or old.get("birth_time_ns") is None
                or old["birth_time_ns"] != ident["birth_time_ns"]
                or (old.get("token") is not None and old["token"] != ident.get("token"))
            ):
                reason = "administrative inode, birth time or generation token differs or is unavailable"
            elif current["scope_error"]:
                reason = "current worktree registration is outside the safe scope"
            else:
                actual, reason = directory_evidence(
                    saved, current, Path(state.data["repo"])
                )
        outcome = "migrated" if actual is not None else "retired"
        if actual is None and (
            residue := unregistered_residue(saved, Path(state.data["repo"]))
        ):
            outcome, reason = "blocked", reason + "; " + residue
        saved["filesystem_migration"] = {
            "result": outcome,
            "checked_at": utc(now),
            "reason": reason,
            "previous_phase": saved["phase"],
            "previous_admin_identity": old,
            "previous_directory_identity": saved.get("directory_identity"),
        }
        if actual is not None and current is not None:
            saved["admin_identity"] = current["admin_identity"]
            saved["directory_identity"] = actual
            saved.pop("registration_lost_at", None)
        elif outcome != "blocked":
            saved.update(phase="superseded", superseded_at=utc(now))
    state.data["schema_version"] = 2
    if changed:
        state.save()
