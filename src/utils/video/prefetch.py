"""Bounded prefetching for producer/consumer video pipelines."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

TItem = TypeVar("TItem")


@dataclass(frozen=True)
class _QueueItem(Generic[TItem]):
    value: TItem | None = None
    error: BaseException | None = None
    done: bool = False


class PrefetchIterator(Generic[TItem]):
    """Prefetch items from an iterable on a bounded background thread."""

    def __init__(self, iterable: Iterable[TItem], *, max_prefetch: int) -> None:
        if max_prefetch <= 0:
            raise ValueError(f"max_prefetch must be positive, got {max_prefetch}")
        self._iterable = iterable
        self._queue: queue.Queue[_QueueItem[TItem]] = queue.Queue(maxsize=max_prefetch)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def __iter__(self) -> Iterator[TItem]:
        if not self._started:
            self._thread.start()
            self._started = True

        while True:
            item = self._queue.get()
            if item.error is not None:
                raise item.error
            if item.done:
                return
            if item.value is None:
                raise RuntimeError("PrefetchIterator received an empty queue item.")
            yield item.value

    def _run(self) -> None:
        try:
            for value in self._iterable:
                self._queue.put(_QueueItem(value=value))
        except BaseException as exc:
            self._queue.put(_QueueItem(error=exc))
        finally:
            self._queue.put(_QueueItem(done=True))
