"""Opt-in inference provenance, independent of model and artifact identity.

See ``inference_recording.md`` for timing/resource semantics and the runtime API.
Importing this module does not import torch/psutil or create a CUDA context.
"""

from __future__ import annotations

import functools
import heapq
import json
import math
import os
import resource
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

STAGES = (
    "node", "clip", "decode", "preprocess", "transfer", "inference",
    "postprocess", "save", "publication", "artifact_load", "model_load",
    "model_unload", "measurement",
)
_GPU_STAGES = {"transfer", "inference", "model_load", "model_unload"}
_ACTIVE: ContextVar[InferenceRecorder | None] = ContextVar("inference_recorder", default=None)
_ATTRIBUTES: ContextVar[dict[str, Any] | None] = ContextVar("inference_recording_attributes", default=None)
_STACK: ContextVar[tuple[int, ...]] = ContextVar("inference_recording_stack", default=())
P = ParamSpec("P")
R = TypeVar("R")


def wall_breakdown(events: Sequence[Mapping[str, Any]], total_ns: int) -> dict[str, float]:
    """Partition wall time: deepest active span wins; GPU wins equal depths.

    Coarse clip/node spans are excluded. Equal-depth CPU ties use the latest
    started span. This attribution convention is not a measure of GPU occupancy.
    """
    edges: list[tuple[int, bool, int]] = []
    for index, event in enumerate(events):
        if event["stage"] not in {"node", "clip"}:
            edges.extend(((int(event["start_ns"]), True, index), (int(event["end_ns"]), False, index)))
    active: set[int] = set()
    heap: list[tuple[int, int, int, int, str]] = []
    duration: dict[str, int] = defaultdict(int)
    previous = 0
    for position, starting, index in sorted(edges):
        while heap and heap[0][3] not in active:
            heapq.heappop(heap)
        duration[heap[0][4] if heap else "unattributed"] += position - previous
        previous = position
        if starting:
            event = events[index]
            stage = str(event["stage"])
            active.add(index)
            heapq.heappush(heap, (-int(event["depth"]), -int(stage in _GPU_STAGES), -position, index, stage))
        else:
            active.discard(index)
    duration["unattributed"] += total_ns - previous
    return {stage: ns / 1e9 for stage, ns in sorted(duration.items())}


class InferenceRecorder:
    """One execution's JSON recorder; use as a context manager exactly once."""

    def __init__(self, destination: Path, *, metadata: Mapping[str, Any] | None = None,
                 devices: Sequence[str] = (), sample_interval: float = 0.1) -> None:
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        self.destination = destination
        self.metadata = dict(metadata or {})
        self.devices = tuple(devices)
        self.sample_interval = sample_interval
        self.events: list[dict[str, Any]] = []
        self.resources: dict[str, Any] = {"ram_rss_sampled_peak_bytes": 0, "temporary_disk_peak_bytes": 0}
        self._assets: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._models: dict[str, dict[str, Any]] = {}
        self._temporary: set[Path] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._sample_errors: list[str] = []
        self._started_ns: int | None = None
        self._next_id = 0

    def __enter__(self) -> InferenceRecorder:
        if _ACTIVE.get() is not None or self._started_ns is not None:
            raise RuntimeError("Recorders cannot be nested or reused")
        import psutil  # type: ignore[import-untyped]

        self._process = psutil.Process()
        self._started_ns = time.perf_counter_ns()
        self._started_at = datetime.now(UTC).isoformat()
        self._token = _ACTIVE.set(self)
        self._stack_token = _STACK.set(())
        try:
            with span("measurement", operation="resource_initialization"):
                if self.devices:
                    import torch

                    torch.cuda.init()
                    for device in self.devices:
                        torch.cuda.reset_peak_memory_stats(device)
            self._sample()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="inference-resources")
            self._thread.start()
        except BaseException:
            _STACK.reset(self._stack_token)
            _ACTIVE.reset(self._token)
            raise
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        try:
            self._finish(kind, error)
        finally:
            _STACK.reset(self._stack_token)
            _ACTIVE.reset(self._token)

    def _finish(self, kind: Any, error: Any) -> None:
        self._stop.set()
        self._thread.join()
        try:
            self._sample()
            self.resources["ram_process_lifetime_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            if self.devices:
                import torch

                self.resources["cuda"] = {device: {
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
                } for device in self.devices}
        except Exception as exc:
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
        assert self._started_ns is not None
        total_ns = time.perf_counter_ns() - self._started_ns
        events = sorted(self.events, key=lambda event: event["id"])
        inclusive: dict[str, float] = defaultdict(float)
        models: dict[str, dict[str, Any]] = {}
        batches: dict[str, Counter[str]] = defaultdict(Counter)
        for event in events:
            seconds = (event["end_ns"] - event["start_ns"]) / 1e9
            inclusive[event["stage"]] += seconds
            attrs = event["attributes"]
            if event["stage"] in {"model_load", "model_unload"}:
                key = json.dumps({k: attrs[k] for k in ("model", "weights", "device", "dtype")}, sort_keys=True)
                model = models.setdefault(key, {**{k: attrs[k] for k in ("model", "weights", "device", "dtype")},
                                               "load_count": 0, "unload_count": 0, "failed_count": 0,
                                               "load_seconds": 0.0, "unload_seconds": 0.0})
                action = event["stage"].removeprefix("model_")
                model[f"{action}_count"] += 1
                model[f"{action}_seconds"] += seconds
                model["failed_count"] += event["status"] != "ok"
            if "batch_shape" in attrs:
                batches[str(attrs.get("model", attrs.get("node", "unknown")))][json.dumps(attrs["batch_shape"])] += 1
        document = {
            "schema": "inference_recording_v1", "started_at": self._started_at,
            "status": "failed" if error is not None or self._sample_errors else "complete",
            "error_type": None if kind is None else kind.__name__, "metadata": self.metadata,
            "timing": {"clock": "perf_counter_ns", "gpu": "synchronized_wall" if self.devices else "not_configured",
                       "wall_attribution": "deepest_span_then_gpu_then_latest", "sample_interval_seconds": self.sample_interval},
            "total_wall_seconds": total_ns / 1e9,
            "stage_wall_seconds": wall_breakdown(events, total_ns),
            "stage_inclusive_seconds": dict(inclusive), "models": list(models.values()),
            "batch_shapes": dict(batches), "resources": self.resources,
            "measurement_errors": self._sample_errors, "events": events,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.destination.with_name(f".{self.destination.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        temporary.replace(self.destination)
        if self._sample_errors and error is None:
            raise RuntimeError(f"Inference resource sampling failed: {self._sample_errors}")

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            try:
                self._sample()
            except Exception as exc:
                self._sample_errors.append(f"{type(exc).__name__}: {exc}")
                return

    def _sample(self) -> None:
        rss = self._process.memory_info().rss
        with self._lock:
            self.resources["ram_rss_sampled_peak_bytes"] = max(self.resources["ram_rss_sampled_peak_bytes"], rss)
            disk = 0
            for root in self._temporary:
                for path in root.rglob("*"):
                    try:
                        if path.is_file():
                            disk += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic publication can remove the sampled temporary path.
            self.resources["temporary_disk_peak_bytes"] = max(self.resources["temporary_disk_peak_bytes"], disk)

    def _synchronize(self, stage: str, device: str = "") -> None:
        if self.devices and (stage in _GPU_STAGES or device.startswith("cuda")):
            import torch

            for device in self.devices:
                torch.cuda.synchronize(device)

    def weight_hashes(self, weights: Mapping[str, Path]) -> dict[str, dict[str, str]]:
        from src.utils.checksum import dual_sha256

        result = {}
        with span("measurement", operation="weight_sha256"):
            for name, path in weights.items():
                path = path.resolve(strict=True)
                info = path.stat()
                signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                with self._lock:
                    cached = self._assets.get(path)
                    if cached is not None and cached[0] != signature:
                        raise ValueError(f"Model weights changed during recording: {path}")
                    if cached is None:
                        cached = (signature, dual_sha256(path))
                        self._assets[path] = cached
                result[name] = {"path": str(path), "sha256": cached[1]}
        return result


@contextmanager
def recording_context(**attributes: Any) -> Iterator[None]:
    """Attach clip/camera/node/pass context, without adding it to results."""
    token = _ATTRIBUTES.set({**(_ATTRIBUTES.get() or {}), **attributes})
    try:
        yield
    finally:
        _ATTRIBUTES.reset(token)


@contextmanager
def span(stage: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Record a completed/error span; yielded attributes may receive outcomes."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield {}
        return
    if stage not in STAGES:
        raise ValueError(f"Unknown inference recording stage: {stage}")
    if "batch_shape" in attributes:
        shape = attributes["batch_shape"]
        if not shape or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("batch_shape must contain nonnegative integer dimensions")
        attributes["batch_shape"] = list(shape)
        attributes.setdefault("batch_items", shape[0])
    stack = _STACK.get()
    with recorder._lock:
        identifier = recorder._next_id
        recorder._next_id += 1
    recorder._synchronize(stage, str(attributes.get("device", "")))
    assert recorder._started_ns is not None
    event: dict[str, Any] = {"id": identifier, "parent_id": stack[-1] if stack else None, "depth": len(stack),
             "thread_id": threading.get_ident(), "stage": stage,
             "start_ns": time.perf_counter_ns() - recorder._started_ns,
             "attributes": {**(_ATTRIBUTES.get() or {}), **attributes}, "status": "ok"}
    token = _STACK.set((*stack, identifier))
    try:
        yield event["attributes"]
    except BaseException as exc:
        event.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        try:
            recorder._synchronize(stage, str(attributes.get("device", "")))
        finally:
            event["end_ns"] = time.perf_counter_ns() - recorder._started_ns
            _STACK.reset(token)
            with recorder._lock:
                recorder.events.append(event)


def measured(stage: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Measure ordinary functions. Generator work must use spans around work."""
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if _ACTIVE.get() is None:
                return function(*args, **kwargs)
            with span(stage, operation=function.__qualname__):
                return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def model_event(action: str, model: str, *, weights: Mapping[str, Path], device: str,
                dtype: str, instance: str) -> Iterator[None]:
    """C3 runtime API: call at actual load/unload, never at model borrowing."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    if action not in {"load", "unload"}:
        raise ValueError("Model event action must be load or unload")
    hashes = recorder.weight_hashes(weights)
    attributes = dict(model=model, weights=hashes, device=device, dtype=dtype, instance=instance)
    with span(f"model_{action}", **attributes):
        yield
    if action == "load":
        recorder._models[instance] = attributes
    else:
        recorder._models.pop(instance, None)


def model_action(action: str, model: str, *, weights: Callable[[Any], Mapping[str, Path]],
                 device: Callable[[Any], str], loaded: Callable[[Any], bool] | None = None
                 ) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Instrument eager constructors or idempotent lifecycle methods in place."""
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if _ACTIVE.get() is None:
                return function(*args, **kwargs)
            owner = args[0]
            if loaded is not None and loaded(owner) == (action == "load"):
                return function(*args, **kwargs)
            with model_event(action, model, weights=weights(owner), device=device(owner),
                             dtype="float32", instance=str(id(owner))):
                return function(*args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def model_release(instance: str) -> Iterator[None]:
    """Record an eager encoder's existing reference-drop/cleanup boundary."""
    recorder = _ACTIVE.get()
    attributes = None if recorder is None else recorder._models.get(instance)
    if attributes is None:
        yield
        return
    with span("model_unload", **attributes):
        yield
    assert recorder is not None
    recorder._models.pop(instance, None)


@contextmanager
def temporary_directory(path: Path) -> Iterator[None]:
    """Sample only explicitly owned temporary output, including its last size."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    with recorder._lock:
        recorder._temporary.add(path)
    try:
        yield
    finally:
        recorder._sample()
        with recorder._lock:
            recorder._temporary.remove(path)
