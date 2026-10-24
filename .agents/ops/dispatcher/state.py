"""Keep a lost ownership registry from becoming a new dispatcher installation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from parent.state import Store

from .schedule import Schedule


class DispatchStore(Store):
    def __init__(self, directory: Path, schedule: Schedule) -> None:
        super().__init__(directory)
        self.schedule = schedule

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
        with super().locked():
            self._require_registry()
            yield self

    def save(self) -> None:
        self._require_registry()
        super().save()
