"""Opt-in inference provenance, independent of model and artifact identity.

See ``inference_recording.md`` for timing/resource semantics and the runtime API.
Importing this module does not import torch/psutil or create a CUDA context.
"""

from __future__ import annotations

import functools
import heapq
import json
import math
import resource
import shutil
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from src.utils.inference_timing import TimingSummary

if TYPE_CHECKING:
    from src.utils.inference_resources import NvmlSampler

STAGES = (
    "node", "clip", "decode", "preprocess", "transfer", "inference",
    "postprocess", "save", "publication", "artifact_load", "model_load",
    "model_unload", "measurement", "video_pass", "video_read", "batch_dispatch",
    "decode_queue_wait", "cuda_sync", "planning", "progress_write",
)
_GPU_STAGES = {"transfer", "inference", "model_load", "model_unload"}
DEFAULT_SAMPLE_INTERVAL = 0.5
_ACTIVE: ContextVar[InferenceRecorder | None] = ContextVar("inference_recorder", default=None)
_ATTRIBUTES: ContextVar[dict[str, Any] | None] = ContextVar("inference_recording_attributes", default=None)
_STACK: ContextVar[tuple[int, ...]] = ContextVar("inference_recording_stack", default=())
_LIFECYCLE: ContextVar[bool] = ContextVar("inference_model_lifecycle", default=False)
P = ParamSpec("P")
R = TypeVar("R")


def _encode_event(event: Mapping[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _chunk_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as stream:
        for row in stream:
            event_id, data = row.rstrip("\n").split("\t", 1)
            yield int(event_id), data


def wall_breakdown(events: Sequence[Mapping[str, Any]], total_ns: int, *, start_ns: int = 0) -> dict[str, float]:
    """Partition wall time: deepest active span wins; GPU wins equal depths.

    Coarse clip/node spans are excluded. Equal-depth CPU ties use the latest
    started span. This attribution convention is not a measure of GPU occupancy.
    """
    edges: list[tuple[int, bool, int]] = []
    for index, event in enumerate(events):
        if event["stage"] not in {"node", "clip", "video_pass"} and int(event["end_ns"]) > int(event["start_ns"]):
            edges.extend(((int(event["start_ns"]), True, index), (int(event["end_ns"]), False, index)))
    active: set[int] = set()
    heap: list[tuple[int, int, int, int, str]] = []
    duration: dict[str, int] = defaultdict(int)
    previous = start_ns
    for position, starting, index in sorted(edges):
        while heap and heap[0][3] not in active:
            heapq.heappop(heap)
        duration[heap[0][4] if heap else "unattributed"] += position - previous
        previous = position
        if starting:
            event = events[index]
            stage = str(event["stage"])
            active.add(index)
            heapq.heappush(heap, (-int(event["depth"]), -int(stage in _GPU_STAGES or stage == "cuda_sync"), -position, index, stage))
        else:
            active.discard(index)
    duration["unattributed"] += total_ns - previous
    return {stage: ns / 1e9 for stage, ns in sorted(duration.items())}


class InferenceRecorder:
    """One execution's JSON recorder; use as a context manager exactly once."""

    def __init__(self, destination: Path | None, *, metadata: Mapping[str, Any] | None = None,
                 devices: Sequence[str] = (), sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
                 synchronize_cuda: bool = True, model_aliases: Mapping[str, str] | None = None) -> None:
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        self.destination = destination
        self.metadata = dict(metadata or {})
        self.devices = tuple(devices)
        self.sample_interval = sample_interval
        self.synchronize_cuda = synchronize_cuda
        self.model_aliases = dict(model_aliases or {})
        self.events: list[dict[str, Any]] = []
        self.resources: dict[str, Any] = {"ram_rss_sampled_peak_bytes": 0, "temporary_disk_peak_bytes": 0}
        self.samples: list[dict[str, Any]] = []
        self._nvml: NvmlSampler | None = None
        self._roles: dict[int, str] = {}
        self._assets: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._models: dict[str, dict[str, Any]] = {}
        self._temporary: set[Path] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._sample_errors: list[str] = []
        self._started_ns: int | None = None
        self._next_id = 0
        self._live_spans = 0
        self._event_directory: Path | None = None
        self._event_chunks: list[tuple[int, Path]] = []
        self._video_pass_spans: list[dict[str, Any]] = []

    def __enter__(self) -> InferenceRecorder:
        if _ACTIVE.get() is not None or self._started_ns is not None:
            raise RuntimeError("Recorders cannot be nested or reused")
        from src.utils.inference_resources import NvmlSampler, ResourceSampler

        self._started_ns = time.perf_counter_ns()
        self._started_at = datetime.now(UTC).isoformat()
        self._token = _ACTIVE.set(self)
        self._stack_token = _STACK.set(())
        self._roles[threading.get_native_id()] = "main"
        try:
            with span("measurement", operation="resource_initialization"):
                if self.devices:
                    import torch

                    torch.cuda.init()
                    for device in self.devices:
                        torch.cuda.reset_peak_memory_stats(device)
                    self._nvml = NvmlSampler({
                        device: str(torch.cuda.get_device_properties(device).uuid) for device in self.devices
                    })
                self._sampler = ResourceSampler(self._started_ns, self._nvml)
            self._sample()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="inference-resources")
            self._thread.start()
        except BaseException as error:
            try:
                self._sample_errors.append(f"{type(error).__name__}: {error}")
                self._close_nvml()
                self._write_record(type(error), error)
            finally:
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
            if not self._sample_errors:
                self._sample()
            if self.devices:
                import torch

                self.resources["cuda"] = {device: {
                    "allocated_peak_bytes": torch.cuda.max_memory_allocated(device),
                    "reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
                } for device in self.devices}
        except Exception as exc:
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            self._close_nvml()
        self._write_record(kind, error)
        if self._sample_errors and error is None:
            raise RuntimeError(f"Inference resource sampling failed: {self._sample_errors}")

    def _close_nvml(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.close()
            except Exception as exc:
                self._sample_errors.append(f"{type(exc).__name__}: {exc}")

    def checkpoint_events(self, *, minimum_events: int = 100_000,
                          consume_pending: Callable[[], None] | None = None) -> bool:
        """Spill completed spans between video-batch nodes or dispatch groups."""
        with self._lock:
            if len(self.events) < minimum_events or self._live_spans or self.destination is None:
                return False
            if consume_pending is not None:
                consume_pending()
            assert self._started_ns is not None
            cutoff_ns = time.perf_counter_ns() - self._started_ns
            events, self.events = self.events, []
        try:
            with span("measurement", operation="checkpoint_events"):
                if self._event_directory is None:
                    self._event_directory = Path(tempfile.mkdtemp(prefix=".recording-", dir=self.destination.parent))
                chunk = self._event_directory / f"{len(self._event_chunks):06}.tsv"
                coarse = []
                with chunk.open("w", encoding="utf-8") as output:
                    for event in sorted(events, key=lambda item: item["id"]):
                        if event["stage"] == "video_pass":
                            coarse.append(event)
                        else:
                            output.write(f"{event['id']}\t{_encode_event(event)}\n")
        except BaseException:
            with self._lock:
                self.events[:0] = events
            raise
        # A video_pass is a coarse envelope that can span multiple windows.
        self._video_pass_spans.extend(coarse)
        self._event_chunks.append((cutoff_ns, chunk))
        return True

    def _event_windows(self, final_events: list[dict[str, Any]], total_ns: int
                       ) -> Iterator[tuple[int | None, list[dict[str, Any]]]]:
        if not self._event_chunks:
            yield total_ns, final_events
            return
        for cutoff_ns, path in self._event_chunks:
            yield cutoff_ns, [json.loads(data) for _, data in _chunk_lines(path)]
        yield total_ns, [event for event in final_events if event["stage"] != "video_pass"]
        yield None, self._video_pass_spans + [event for event in final_events if event["stage"] == "video_pass"]

    def _raw_events(self, final_events: Sequence[dict[str, Any]]) -> Iterator[str]:
        saved = (item for _, path in self._event_chunks for item in _chunk_lines(path))
        remaining = ((event["id"], _encode_event(event))
                     for event in sorted([*self._video_pass_spans, *final_events], key=lambda item: item["id"]))
        for _, data in heapq.merge(saved, remaining):
            yield data

    def _write_record(self, kind: Any, error: Any) -> None:
        if self.destination is None:
            return
        assert self._started_ns is not None
        total_ns = time.perf_counter_ns() - self._started_ns
        final_events = sorted(self.events, key=lambda event: event["id"])
        inclusive: dict[str, float] = defaultdict(float)
        models: dict[str, dict[str, Any]] = {}
        batches: dict[str, Counter[str]] = defaultdict(Counter)
        dispatches: dict[str, dict[str, Any]] = {}
        video_reads: dict[str, Counter[str]] = defaultdict(Counter)
        pass_loads: dict[str, Counter[str]] = defaultdict(Counter)
        attributed: dict[str, float] = defaultdict(float)
        timeline = TimingSummary(self.samples, self.model_aliases)
        start_ns = 0
        for cutoff_ns, events in self._event_windows(final_events, total_ns):
            if cutoff_ns is not None:
                for stage, seconds in wall_breakdown(events, cutoff_ns, start_ns=start_ns).items():
                    attributed[stage] += seconds
                start_ns = cutoff_ns
            timeline.add(events)
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
                    if action == "load" and "video_pass" in attrs:
                        pass_loads[str(attrs["video_pass"])][attrs["model"]] += 1
                if "batch_shape" in attrs:
                    batches[str(attrs.get("model", attrs.get("node", "unknown")))][json.dumps(attrs["batch_shape"])] += 1
                if event["stage"] == "batch_dispatch":
                    stats = dispatches.setdefault(attrs["model"], {"batches": 0, "failed": 0, "items": 0,
                                                 "total_capacity": 0, "sizes": Counter(), "cross_source_batches": 0,
                                                 "cross_clip_batches": 0, "fill_fraction": 0.0,
                                                 "peak_queued_items": 0, "peak_active_sources": 0})
                    stats["batches"] += 1
                    stats["failed"] += event["status"] != "ok"
                    stats["items"] += attrs["batch_items"]
                    stats["total_capacity"] += attrs["batch_capacity"]
                    stats["sizes"][str(attrs["batch_items"])] += 1
                    stats["cross_source_batches"] += attrs["source_count"] > 1
                    stats["cross_clip_batches"] += attrs["clip_count"] > 1
                    stats["fill_fraction"] = stats["items"] / stats["total_capacity"]
                    stats["peak_queued_items"] = max(stats["peak_queued_items"], attrs["queued_items"])
                    stats["peak_active_sources"] = max(stats["peak_active_sources"], attrs["active_sources"])
                if event["stage"] == "video_read":
                    component = str(attrs.get("node", "unknown")).split("/")[0]
                    video_reads[str(attrs.get("video_pass", "outside_plan"))][component] += 1
        self.resources["ram_process_lifetime_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        document = {
            "schema": "inference_recording_v1", "started_at": self._started_at,
            "status": "failed" if error is not None or self._sample_errors else "complete",
            "error_type": None if kind is None else kind.__name__, "metadata": self.metadata,
            "timing": {"clock": "perf_counter_ns", "gpu": self._gpu_timing(),
                       "wall_attribution": "deepest_span_then_gpu_then_latest", "sample_interval_seconds": self.sample_interval,
                       "event_checkpoints": len(self._event_chunks)},
            "total_wall_seconds": total_ns / 1e9,
            "stage_wall_seconds": dict(sorted(attributed.items())),
            "stage_inclusive_seconds": dict(inclusive), "models": list(models.values()),
            "batch_shapes": dict(batches), "batch_dispatches": dispatches, "resources": self.resources,
            "video_reads_by_pass": video_reads, "model_loads_by_pass": pass_loads,
            "resource_samples": self.samples, "timings": timeline.result(),
            "measurement_errors": self._sample_errors,
        }
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write("{")
                for key, value in document.items():
                    stream.write(f"{json.dumps(key)}:")
                    json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    stream.write(",")
                stream.write('"events":[')
                for index, data in enumerate(self._raw_events(final_events)):
                    if index:
                        stream.write(",")
                    stream.write(data)
                stream.write("]}\n")
                stream.close()
                temporary.replace(self.destination)
            finally:
                temporary.unlink(missing_ok=True)
        if self._event_directory is not None:
            shutil.rmtree(self._event_directory)

    def _gpu_timing(self) -> str:
        if not self.devices:
            return "not_configured"
        return "synchronized_wall" if self.synchronize_cuda else "host_wall"

    def _sample_loop(self) -> None:
        with self._lock:
            self._roles[threading.get_native_id()] = "resource_sampler"
        while not self._stop.wait(self.sample_interval):
            try:
                self._sample()
            except Exception as exc:
                self._sample_errors.append(f"{type(exc).__name__}: {exc}")
                return

    def _sample(self) -> None:
        with self._lock:
            roles = dict(self._roles)
        sample = self._sampler.sample(roles)
        self.samples.append(sample)
        self.resources["ram_rss_sampled_peak_bytes"] = max(
            self.resources["ram_rss_sampled_peak_bytes"], sample["rss_bytes"])
        minimum = self.resources.get("system_memory_available_min_bytes", sample["system_memory_available_bytes"])
        self.resources["system_memory_available_min_bytes"] = min(minimum, sample["system_memory_available_bytes"])
        self.resources["system_memory_total_bytes"] = sample["system_memory_total_bytes"]
        for device, gpu in sample["gpu"].items():
            peak = self.resources.setdefault("nvml", {}).setdefault(device, {
                "used_peak_bytes": 0, "total_bytes": gpu["total_bytes"]})
            peak["used_peak_bytes"] = max(peak["used_peak_bytes"], gpu["used_bytes"])
        self._sample_temporary()

    def _sample_temporary(self) -> None:
        from src.utils.inference_resources import temporary_bytes

        with self._lock:
            disk = temporary_bytes(tuple(self._temporary))
            self.resources["temporary_disk_peak_bytes"] = max(self.resources["temporary_disk_peak_bytes"], disk)

    def _synchronize(self, stage: str, boundary: str, attributes: Mapping[str, Any]) -> None:
        device = str(attributes.get("device", ""))
        if not self.devices or not self.synchronize_cuda or stage == "cuda_sync":
            return
        if not (device.startswith("cuda") or not device and stage in _GPU_STAGES):
            return
        import torch

        labels = {"model": attributes["model"]} if "model" in attributes else {}
        with span("cuda_sync", operation="timing", boundary=boundary, awaited_stage=stage, **labels):
            for cuda_device in self.devices:
                torch.cuda.synchronize(cuda_device)

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
def recording_thread(role: str) -> Iterator[None]:
    """Label the OS thread in spans and resource samples."""
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
        return
    tid = threading.get_native_id()
    with recorder._lock:
        previous = recorder._roles.get(tid)
        recorder._roles[tid] = role
    try:
        yield
    finally:
        with recorder._lock:
            if previous is None:
                recorder._roles.pop(tid, None)
            else:
                recorder._roles[tid] = previous


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
    native_thread_id = threading.get_native_id()
    with recorder._lock:
        identifier = recorder._next_id
        recorder._next_id += 1
        if stage != "video_pass":
            recorder._live_spans += 1
        role = recorder._roles.get(native_thread_id, "native")
    assert recorder._started_ns is not None
    event: dict[str, Any] = {"id": identifier, "parent_id": stack[-1] if stack else None, "depth": len(stack),
             "thread_id": threading.get_ident(), "native_thread_id": native_thread_id, "thread_role": role, "stage": stage,
             "start_ns": time.perf_counter_ns() - recorder._started_ns,
             "thread_cpu_start_ns": time.thread_time_ns(), "process_cpu_start_ns": time.process_time_ns(),
             "attributes": {**(_ATTRIBUTES.get() or {}), **attributes}, "status": "ok"}
    token = _STACK.set((*stack, identifier))
    try:
        recorder._synchronize(stage, "before", event["attributes"])
        yield event["attributes"]
        recorder._synchronize(stage, "after", event["attributes"])
    except BaseException as exc:
        event.update(status="error", error_type=type(exc).__name__)
        raise
    finally:
        event["thread_cpu_ns"] = time.thread_time_ns() - event["thread_cpu_start_ns"]
        event["process_cpu_ns"] = time.process_time_ns() - event["process_cpu_start_ns"]
        event["end_ns"] = time.perf_counter_ns() - recorder._started_ns
        _STACK.reset(token)
        with recorder._lock:
            recorder.events.append(event)
            if stage != "video_pass":
                recorder._live_spans -= 1


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
    if recorder is None or _LIFECYCLE.get():
        yield
        return
    if action not in {"load", "unload"}:
        raise ValueError("Model event action must be load or unload")
    hashes = recorder.weight_hashes(weights)
    attributes = dict(model=model, weights=hashes, device=device, dtype=dtype, instance=instance)
    token = _LIFECYCLE.set(True)
    try:
        with span(f"model_{action}", **attributes):
            yield
    finally:
        _LIFECYCLE.reset(token)
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
        try:
            recorder._sample_temporary()
        finally:
            with recorder._lock:
                recorder._temporary.remove(path)
