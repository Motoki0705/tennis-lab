"""Bounded inference work across the ready sources of one component stage."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Hashable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from src.utils.inference_recording import recording_context, span

if TYPE_CHECKING:
    from src.tennis_scene.pipeline.inference_runtime import ModelBorrower, NodeKey


@dataclass(frozen=True, slots=True)
class BatchSizes:
    court: int = 2
    dino: int = 2
    vitpose: int = 4
    clip: int = 16
    hmr2: int = 4
    ball: int = 1

    def __post_init__(self) -> None:
        for name in self.__slots__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"batching.sizes.{name} must be a positive integer")

    def for_model(self, name: str) -> int:
        if name not in self.__slots__:
            raise ValueError(f"Undeclared batch size for {name}")
        return int(getattr(self, name))


@dataclass(frozen=True, slots=True)
class BatchConfig:
    enabled: bool = False
    sizes: BatchSizes = field(default_factory=BatchSizes)
    max_sources: int = 4
    max_pending_items: int = 32
    max_working_bytes: int = 3 * 1024**3
    store_cache_bytes: int = 0
    clips_per_pass: int | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("batching.enabled must be a boolean")
        for name in ("max_sources", "max_pending_items", "max_working_bytes", "store_cache_bytes", "clips_per_pass"):
            value = getattr(self, name)
            if name == "clips_per_pass" and value is None:
                continue
            if type(value) is not int or value < (0 if name == "store_cache_bytes" else 1):
                raise ValueError(f"Invalid batching.{name}: {value}")
        if self.max_pending_items < max(self.sizes.for_model(name) for name in self.sizes.__slots__):
            raise ValueError("batching.max_pending_items must cover every configured batch size")


class BatchMemoryLimit(RuntimeError):
    """An explicit work-set reservation could not be satisfied."""


@dataclass
class WorkingMemory:
    limit: int
    used: int = field(init=False, default=0)
    peak: int = field(init=False, default=0)

    def reserve(self, size: int) -> None:
        if size < 0:
            raise ValueError("A memory reservation cannot be negative")
        if self.used + size > self.limit:
            raise BatchMemoryLimit(
                f"Inference working set needs {self.used + size:,} bytes, exceeding batching.max_working_bytes={self.limit:,}"
            )
        self.used += size
        self.peak = max(self.peak, self.used)

    def release(self, size: int) -> None:
        self.used -= size
        if self.used < 0:
            raise RuntimeError("Unbalanced inference memory reservation")


@dataclass
class SourceMemory:
    budget: WorkingMemory
    size: int = field(init=False, default=0)

    def reserve(self, size: int) -> None:
        self.budget.reserve(size)
        self.size += size

    def close(self) -> None:
        self.budget.release(self.size)
        self.size = 0

    @contextmanager
    def workspace(self, size: int) -> Iterator[None]:
        self.budget.reserve(size)
        try:
            yield
        finally:
            self.budget.release(size)


@dataclass(frozen=True, slots=True)
class BatchAddress:
    clip_id: str
    camera_id: str
    source_frame: int
    detection_row: int | None = None
    person_id: int | None = None
    segment: int | None = None
    window_end: int | None = None  # Exclusive bound on real frames before tail handling.


@dataclass(frozen=True, slots=True)
class BatchUnit:
    address: BatchAddress
    compatibility: Hashable
    payload: Any
    working_bytes: int


@dataclass(frozen=True)
class BatchSource:
    units: Generator[BatchUnit, None, None]
    write: Callable[[BatchUnit, Any], None]
    finish: Callable[[], Any]


class BatchAdapter(Protocol):
    """A component owns its inputs, forward pass and per-source writeback."""

    @property
    def batch_model(self) -> str: ...

    @property
    def batch_alias(self) -> str: ...

    def open_batch(self, clip_id: str, inputs: Any, models: ModelBorrower, memory: SourceMemory) -> BatchSource: ...

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]: ...


@dataclass
class BatchJob:
    key: NodeKey
    camera_id: str | None
    adapter: BatchAdapter
    models: ModelBorrower
    source: BatchSource | None
    memory: SourceMemory
    pending: int = 0
    exhausted: bool = False

    def close(self) -> None:
        source, self.source = self.source, None
        if source is not None:
            try:
                source.units.close()
            finally:
                self.memory.close()


def drain_batch_jobs(
    jobs: Sequence[BatchJob],
    config: BatchConfig,
    budget: WorkingMemory,
    *,
    complete: Callable[[NodeKey, Any], None],
    fail: Callable[[NodeKey, Exception], None],
) -> None:
    """Round-robin sources, buffering only compatible units in a finite queue."""
    if not jobs:
        return
    model = jobs[0].adapter.batch_model
    capacity = config.sizes.for_model(model)
    active = {job.key: job for job in jobs}
    readers = deque(jobs)
    queues: dict[Hashable, list[tuple[BatchJob, BatchUnit]]] = {}
    pending = 0

    def retire(job: BatchJob) -> None:
        if active.pop(job.key, None) is not None:
            job.close()

    def reject(job: BatchJob, error: Exception) -> None:
        nonlocal pending
        for other in tuple(active.values()):
            if other.key[0] == job.key[0]:
                retire(other)
        fail(job.key, error)
        for key, queue in tuple(queues.items()):
            surviving = [(owner, unit) for owner, unit in queue if owner.key in active]
            budget.release(sum(unit.working_bytes for owner, unit in queue if owner.key not in active))
            pending -= len(queue) - len(surviving)
            if surviving:
                queues[key] = surviving
            else:
                del queues[key]

    def finish(job: BatchJob) -> None:
        assert job.source is not None
        with recording_context(clip_id=job.key[0], camera_id=job.camera_id, node=job.key[1]):
            try:
                complete(job.key, job.source.finish())
            except Exception as error:
                reject(job, error)
            else:
                retire(job)

    def flush(key: Hashable) -> None:
        nonlocal pending
        queue = queues.pop(key)
        pending -= len(queue)
        first = queue[0][0]
        try:
            units = [unit for _, unit in queue]
            sources = {(unit.address.clip_id, unit.address.camera_id) for unit in units}
            with span("batch_dispatch", model=model, batch_items=len(units), batch_capacity=capacity,
                      fill_fraction=len(units) / capacity, source_count=len(sources),
                      clip_count=len({clip for clip, _ in sources}), queued_items=pending + len(queue),
                      active_sources=len(active)):
                results = first.adapter.infer_batch(units, first.models)
                if len(results) != len(units):
                    raise RuntimeError("Batch adapter changed the inference unit count")
            for (job, unit), result in zip(queue, results, strict=True):
                if job.key not in active:
                    continue
                assert job.source is not None
                job.pending -= 1
                try:
                    if isinstance(result, Exception):
                        raise result
                    job.source.write(unit, result)
                except Exception as error:
                    reject(job, error)
                else:
                    if job.exhausted and not job.pending:
                        finish(job)
        finally:
            budget.release(sum(unit.working_bytes for _, unit in queue))

    try:
        while readers:
            job = readers.popleft()
            if job.key not in active:
                continue
            assert job.source is not None
            try:
                with recording_context(clip_id=job.key[0], camera_id=job.camera_id, node=job.key[1]):
                    unit = next(job.source.units)
            except StopIteration:
                job.exhausted = True
                if not job.pending:
                    finish(job)
                continue
            except Exception as error:
                reject(job, error)
                continue
            if unit.address.clip_id != job.key[0] or (
                job.camera_id is not None and unit.address.camera_id != job.camera_id
            ):
                raise RuntimeError(f"Inference unit belongs to another source: {unit.address}")
            budget.reserve(unit.working_bytes)
            readers.append(job)
            queue = queues.setdefault(unit.compatibility, [])
            queue.append((job, unit))
            job.pending += 1
            pending += 1
            if len(queue) == capacity:
                flush(unit.compatibility)
            elif pending == config.max_pending_items:
                flush(max(queues, key=lambda key: len(queues[key])))
            del unit, queue
        while queues:
            flush(next(iter(queues)))
    finally:
        for job in jobs:
            retire(job)
        budget.release(sum(unit.working_bytes for queue in queues.values() for _, unit in queue))
