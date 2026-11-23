"""Bounded prefetching for producer/consumer video pipelines."""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator, Iterable
from contextlib import suppress
from contextvars import copy_context
from dataclasses import dataclass
from typing import Generic, TypeVar

from src.utils.inference_recording import recording_error, recording_thread, span

TItem = TypeVar("TItem")
_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _QueueItem(Generic[TItem]):
    value: TItem | None = None
    error: BaseException | None = None
    done: bool = False


class PrefetchIterator(Generic[TItem]):
    """Prefetch items once on a bounded thread stopped by closing the iterator."""

    def __init__(self, iterable: Iterable[TItem], *, max_prefetch: int) -> None:
        if max_prefetch <= 0:
            raise ValueError(f"max_prefetch must be positive, got {max_prefetch}")
        self._iterable = iterable
        self._queue: queue.Queue[_QueueItem[TItem]] = queue.Queue(maxsize=max_prefetch)
        self._stop = threading.Event()
        context = copy_context()
        self._thread = threading.Thread(target=context.run, args=(self._run,), daemon=True, name="decode-prefetch")
        self._started = False

    def __iter__(self) -> Generator[TItem, None, None]:
        if self._started:
            raise RuntimeError("PrefetchIterator can only be consumed once.")
        self._thread.start()
        self._started = True
        try:
            while True:
                with span("decode_queue_wait", operation="read"):
                    item = self._queue.get()
                if item.error is not None:
                    raise item.error
                if item.done:
                    return
                if item.value is None:
                    raise RuntimeError("PrefetchIterator received an empty queue item.")
                yield item.value
        finally:
            self._stop.set()
            # There is one producer and it checks _stop before every write.
            # One free slot is enough to release a concurrent queue.put.
            with suppress(queue.Empty):
                self._queue.get_nowait()
            self._thread.join(timeout=_STOP_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                message = "Decode prefetch producer did not stop after cancellation"
                recording_error(message)
                raise TimeoutError(message)

    def _write(self, value: TItem) -> bool:
        if self._stop.is_set():
            return False
        with span("decode_queue_wait", operation="write") as attributes:
            self._queue.put(_QueueItem(value=value))
            if self._stop.is_set():
                attributes["outcome"] = "interrupted"
                return False
        return True

    def _run(self) -> None:
        with recording_thread("decode_prefetch"):
            terminal: _QueueItem[TItem] = _QueueItem(done=True)
            try:
                source = iter(self._iterable)
                try:
                    while not self._stop.is_set():
                        try:
                            value = next(source)
                        except StopIteration:
                            break
                        if not self._write(value):
                            return
                finally:
                    close = getattr(source, "close", None)
                    if close is not None:
                        close()
            except BaseException as exc:
                terminal = _QueueItem(error=exc)
            if not self._stop.is_set():
                self._queue.put(terminal)
