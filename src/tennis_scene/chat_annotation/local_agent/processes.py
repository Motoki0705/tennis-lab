"""Compatibility imports until the legacy workflow is removed by the integration change."""

from src.tasks.base.annotation.workers.processes import (
    ProcessIdentity,
    capture_supervisor,
    owned_members,
    process_snapshot,
    require_process_backend,
    signal_owned,
)

__all__ = [
    "ProcessIdentity",
    "capture_supervisor",
    "owned_members",
    "process_snapshot",
    "require_process_backend",
    "signal_owned",
]
