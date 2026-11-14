"""Keep a lost ownership registry from becoming a new dispatcher installation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parent.github import GitHub
from parent.state import Store

if TYPE_CHECKING:
    from .schedule import Schedule


def require_client(gh: GitHub, store: Store) -> None:
    if gh._store is not store:
        raise ValueError("GitHub client must use the transaction's Store instance")


def client_for(gh: GitHub, store: Store) -> GitHub:
    """An independent parent/query scope, bound to the same transaction."""
    import copy

    require_client(gh, store)
    client = copy.copy(gh)
    client.parent = None
    client._cache = {}
    client._history_rows = None
    client._history_scope = False
    return client


@contextmanager
def repository_scope(store: Store) -> Iterator[None]:
    """Collections inspect multiple owners; never merge against one parent."""
    previous, store._focus = store._focus, None
    try:
        yield
    finally:
        store._focus = previous


class RepositoryStore(Store):
    """Operations whose admission depends on more than one parent."""

    def focus(self, key: str) -> None:
        self._focus = None


class DispatchStore(RepositoryStore):
    def __init__(self, directory: Path, schedule: Schedule) -> None:
        super().__init__(directory)
        self.schedule = schedule
        self._job_observations: list[dict[str, Any]] = []

    @contextmanager
    def unlocked_read(self) -> Iterator[None]:
        self.require_transaction()
        before = self.schedule.read() if self.schedule.path.exists() else None
        try:
            with super().unlocked_read():
                yield
        finally:
            self.require_transaction()
            try:
                after = self.schedule.read() if self.schedule.path.exists() else None
            except (ValueError, OSError, KeyError, TypeError):
                self.conflict("dispatcher schedule could not be revalidated")
            if before != after:
                self.conflict(
                    "dispatcher schedule changed during I/O; retry the command"
                )

    def _require_registry(self) -> None:
        scheduled = self.schedule.path.exists() or self.schedule.path.is_symlink()
        workers = self.schedule.directory / "workers"
        judgements = self.schedule.directory / "judgements"
        worker_records = workers.exists() and any(workers.iterdir())
        judgement_records = judgements.exists() and any(judgements.iterdir())
        missing = not (self.path.exists() or self.path.is_symlink())
        assigned = scheduled and any(
            candidate.get("state") == "assigned"
            for run in self.schedule.read()["runs"]
            for candidate in run["candidates"].values()
        )
        if (missing and (scheduled or worker_records or judgement_records)) or (
            not self.parents and (worker_records or assigned)
        ):
            raise ValueError(
                "parent registry missing or empty despite dispatcher history; "
                "保存・割当・起動を停止しました。dispatcher/README.md の "
                "「registry欠落からの復旧」に従って復元・照合してください"
            )

    @contextmanager
    def locked(self) -> Iterator[Store]:
        observations: list[dict[str, Any]] = []
        try:
            with super().locked():
                try:
                    self._focus = None
                    self._require_registry()
                    yield self
                finally:
                    observations, self._job_observations = self._job_observations, []
        finally:
            if observations:
                from decision_logs.observations import saved_jobs

                saved_jobs(observations)

    def save(self) -> None:
        self._require_registry()
        changed = []
        for key, row in self.parents.items():
            previous = self._saved.get("parents", {}).get(key, {}).get("dispatch") or {}
            for identifier, job in (row.get("dispatch") or {}).get("jobs", {}).items():
                if job != previous.get("jobs", {}).get(identifier):
                    changed.append(deepcopy(job))
        super().save()
        self._job_observations.extend(changed)
