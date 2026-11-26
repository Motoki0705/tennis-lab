"""Describe failures in the journal using only public recovery information."""

from __future__ import annotations

import errno
import sys

from lib.checkout_lock import CheckoutUnavailable
from lib.state_files import StateFileError
from parent.state import RegistryStateError

README = ".agents/ops/dispatcher/README.md"
STOP = "Stop the dispatcher with `spin ops off dispatcher`."
CHECK_STATE = "Check `spin ops status dispatcher` before restarting."
RESTORE_STATE = (
    f"{STOP} Confirm writers have stopped. Inspect the entry without following links, "
    f"preserve it, and restore a verified backup. {CHECK_STATE}"
)

_STATE_PATHS = {
    "schedule.json": "dispatcher/schedule.json",
    "registry.json": "parent-sessions/registry.json",
    "registry.lock": "parent-sessions/registry.lock",
    "registry-v1.backup.json": "parent-sessions/registry-v1.backup.json",
    "judgement.lock": "dispatcher/judgement.lock",
    "supervisor.lock": "dispatcher/supervisor.lock",
    "judgements": "dispatcher/judgements",
    "workers": "dispatcher/workers",
    "index.json": "agent-input/index.json",
    "start-input.json": "worker start-input.json",
    "input.json": "private input.json",
    "result.json": "private result.json",
}
_STATE_REASONS = {
    "symlink": "is a symbolic link",
    "not_regular": "is not a regular file",
    "not_directory": "is not a directory",
    "wrong_owner": "is owned by a different user",
    "hardlink": "has multiple hard links",
    "file_unlinked": "was unlinked during state access",
    "shared_write": "is writable by other users",
    "directory_changed": "was replaced during state access",
    "lock_changed": "was replaced while acquiring or holding the lock",
    "file_changed": "changed during registry rollback",
    "temporary_collision": "already exists",
    "temporary_changed": "changed during the write",
    "access_failed": "could not be accessed",
    "write_failed": "could not be written",
}
_REGISTRY_REASONS = {
    "registry_json": "contains malformed JSON",
    "registry_format": "has an unsupported or invalid registry format",
    "registry_mismatch": "disagrees with its parent records",
    "registry_record_missing": "is missing a parent record listed in the index",
}


class DispatcherError(ValueError):
    """A known failure whose message and recovery are safe for the journal."""

    def __init__(
        self,
        reason: str,
        diagnostic: str,
        recovery: str,
        *,
        state_path: str | None = None,
    ) -> None:
        super().__init__(f"{diagnostic} {recovery}")
        self.report = {"reason": reason, "diagnostic": diagnostic, "recovery": recovery}
        if state_path is not None:
            self.report["state_path"] = state_path


def state_failure(exc: StateFileError) -> DispatcherError:
    path = _STATE_PATHS.get(exc.path.name, "private state file")
    if exc.entry == "legacy_temporary":
        path += ".tmp"
    elif exc.entry == "temporary":
        path = f"Temporary file for {path}"
    elif exc.entry == "directory":
        path = f"Directory containing {path}"
    description = f"{path} {_STATE_REASONS[exc.reason]}"
    code = errno.errorcode.get(exc.errno or 0)
    if code:
        description += f" ({code})"
    if exc.entry in {"legacy_temporary", "temporary"}:
        recovery = (
            f"{STOP} Inspect the temporary entry without following links. "
            f"Move it aside after confirming writers have stopped. {CHECK_STATE}"
        )
    elif exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        recovery = (
            "Free space or resolve the quota for the state filesystem, then retry."
        )
    elif exc.reason in {"access_failed", "write_failed"}:
        recovery = "Check the state filesystem, directory ownership and write permissions, then retry."
    elif exc.entry == "directory":
        recovery = (
            f"{STOP} Verify the configured state directory and its ancestors. Use real "
            f"directories with trusted owners and private write permissions. See {README}."
        )
    elif exc.path.suffix == ".lock":
        recovery = (
            f"{STOP} Inspect the lock entry and identify its holder. Confirm all holders "
            f"have exited before repairing the state directory. See {README}."
        )
    else:
        recovery = f"{RESTORE_STATE} Verify ownership and private permissions."
    return DispatcherError(exc.reason, description + ".", recovery, state_path=path)


def report_failure(exc: Exception) -> dict[str, str]:
    if isinstance(exc, StateFileError):
        exc = state_failure(exc)
    elif isinstance(exc, RegistryStateError):
        path = _STATE_PATHS.get(exc.path.name, "parent-sessions/parents record")
        exc = DispatcherError(
            exc.reason,
            f"{path} {_REGISTRY_REASONS[exc.reason]}.",
            f"{STOP} Preserve the registry and records. Follow the registry recovery "
            f"procedure in {README} to verify and restore a backup.",
            state_path=path,
        )
    elif isinstance(exc, CheckoutUnavailable):
        exc = DispatcherError(
            "checkout_unavailable",
            "The dispatcher checkout configured in dispatcher/schedule.json is unavailable.",
            f"{STOP} Verify the configured checkout. Restore it at the configured location "
            f"or repair the saved path before restarting. See {README}.",
            state_path="dispatcher/schedule.json",
        )
    if not isinstance(exc, DispatcherError):
        exc = DispatcherError(
            "unclassified_error",
            "Dispatcher input, state or runtime error.",
            "Inspect private records.",
        )
    print(f"{exc.report['diagnostic']} {exc.report['recovery']}", file=sys.stderr)
    return {"result": "failed", **exc.report}
