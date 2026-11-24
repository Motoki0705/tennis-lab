"""Synchronized, single-GPU inference comparisons on pinned inputs."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from importlib import metadata
from itertools import cycle
from math import floor, isfinite
from pathlib import Path
from subprocess import CalledProcessError
from time import perf_counter_ns
from typing import Generic, TypeVar

import torch

from src.utils.checksum import dual_sha256
from src.utils.device import DeviceSelectionError, resolve_device
from src.utils.inference_resources import NvmlSampler, _NvmlMemory

__all__ = [
    "Backend",
    "Batch",
    "BatchStages",
    "BenchmarkAdapter",
    "BenchmarkCase",
    "BenchmarkProtocol",
    "BenchmarkReport",
    "ConditionUnavailable",
    "Conversion",
    "ConversionResult",
    "ParityResult",
    "Stage",
    "Status",
    "benchmark",
    "measure_conversion",
]

Input = TypeVar("Input")
Output = TypeVar("Output")
Value = TypeVar("Value")


class Backend(StrEnum):
    PT32 = "PT32"
    PT16 = "PT16"
    TRT16 = "TRT16"


class Stage(StrEnum):
    PREPROCESS = "preprocess"
    H2D = "h2d"
    FORWARD = "forward"
    POSTPROCESS = "postprocess"
    D2H = "d2h"


class Conversion(StrEnum):
    ONNX_EXPORT = "onnx_export"
    ENGINE_BUILD = "engine_build"


class Status(StrEnum):
    OK = "ok"
    OOM = "oom"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    CONVERSION_FAILED = "conversion_failed"
    BACKEND_MISMATCH = "backend_mismatch"
    PARITY_UNAVAILABLE = "parity_unavailable"
    PARITY_FAILED = "parity_failed"


class ConditionUnavailable(RuntimeError):
    """Report a known condition-specific failure with its original evidence."""

    def __init__(self, status: Status, message: str) -> None:
        if status is Status.OK or not message:
            raise ValueError(
                "An unavailable condition requires a failure status and reason"
            )
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class BenchmarkProtocol:
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    warmup: int = 30
    iterations_per_block: int = 100
    blocks: int = 3

    def __post_init__(self) -> None:
        if (
            not self.batch_sizes
            or self.batch_sizes[0] != 1
            or self.batch_sizes != tuple(sorted(set(self.batch_sizes)))
        ):
            raise ValueError("Sweep sizes must be increasing and start with batch 1")
        counts = (
            *self.batch_sizes,
            self.warmup,
            self.iterations_per_block,
            self.blocks,
        )
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError(
                "Sweep sizes and iteration counts must be positive integers"
            )


_DEFAULT_PROTOCOL = BenchmarkProtocol()


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    backend: Backend
    batch_size: int
    item_unit: str
    input_manifest_sha256: str
    settings: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        digest = self.input_manifest_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("The input manifest requires a lowercase SHA-256 digest")
        if not self.name or not self.item_unit or type(self.batch_size) is not int:
            raise ValueError(
                "Report the workload name, item unit and integer batch size"
            )


@dataclass(frozen=True)
class Batch(Generic[Input]):
    ids: tuple[str, ...]
    data: Input
    items: int | None = None
    processed_items: int | None = None

    def __post_init__(self) -> None:
        items = self.item_count
        if not self.ids or not all(self.ids) or type(items) is not int or items < 1:
            raise ValueError("Batches require input IDs and a positive item count")
        if self.processed_items is not None and (
            type(self.processed_items) is not int or self.processed_items < items
        ):
            raise ValueError("Processed items must include every counted item")

    @property
    def item_count(self) -> int:
        return len(self.ids) if self.items is None else self.items


@dataclass(frozen=True)
class BenchmarkAdapter(Generic[Input, Output]):
    backend: Backend
    load: Callable[[], Callable[[Input, BatchStages], Output]]
    weights: Mapping[str, Path]
    engines: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class ParityResult:
    passed: bool
    metrics: Mapping[str, object]
    reason: str | None = None


@dataclass(frozen=True)
class Failure:
    status: Status
    phase: str
    message: str


@dataclass(frozen=True)
class ConversionResult:
    phase: Conversion
    wall_ms: float | None
    failure: Failure | None = None


@dataclass(frozen=True)
class StageSample:
    stage: Stage
    name: str
    wall_ms: float
    gpu_ms: float | None


@dataclass(frozen=True)
class BatchSample:
    block: int
    ids: tuple[str, ...]
    items: int
    processed_items: int | None
    wall_ms: float
    stages: tuple[StageSample, ...]


@dataclass(frozen=True)
class TimingStats:
    samples: int
    items: int
    total_ms: float
    p50_ms: float
    p95_ms: float
    p50_ms_per_item: float
    p95_ms_per_item: float
    items_per_second: float


@dataclass
class BenchmarkReport:
    case: BenchmarkCase
    protocol: BenchmarkProtocol
    conversion: tuple[ConversionResult, ...]
    environment: dict[str, object] = field(default_factory=dict)
    weights: dict[str, dict[str, object]] = field(default_factory=dict)
    engines: dict[str, dict[str, object]] = field(default_factory=dict)
    setup_wall_ms: dict[str, float] = field(default_factory=dict)
    samples: list[BatchSample] = field(default_factory=list)
    nvml_used_bytes: dict[str, int] = field(default_factory=dict)
    torch_peak_allocated_bytes: int | None = None
    torch_peak_reserved_bytes: int | None = None
    parity: ParityResult | None = None
    failure: Failure | None = None
    summary: dict[str, TimingStats] | None = None

    @property
    def status(self) -> Status:
        return Status.OK if self.failure is None else self.failure.status

    def to_dict(self) -> dict[str, object]:
        peak = max(self.nvml_used_bytes.values(), default=None)
        initial = self.nvml_used_bytes.get("before_load")
        return {
            "schema_version": 1,
            "status": self.status,
            **asdict(self),
            "nvml_device_peak_sampled_bytes": peak,
            "nvml_peak_minus_before_load_bytes": (
                None if peak is None or initial is None else peak - initial
            ),
        }


def _wall_ms(start: int) -> float:
    return (perf_counter_ns() - start) / 1e6


def _failure(error: Exception, phase: str, default: Status) -> Failure:
    status = default
    if isinstance(error, torch.OutOfMemoryError):
        status = Status.OOM
    elif isinstance(error, ConditionUnavailable):
        status = error.status
    return Failure(status, phase, f"{type(error).__name__}: {error}")


def _device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("The inference benchmark requires an explicit CUDA device")
    try:
        return resolve_device(target)
    except DeviceSelectionError as error:
        raise ConditionUnavailable(Status.UNAVAILABLE, str(error)) from error


def measure_conversion(
    phase: Conversion,
    operation: Callable[[], object],
    *,
    device: str | torch.device | None = None,
) -> ConversionResult:
    """Time an ONNX export or engine build in its own process."""
    if phase is Conversion.ENGINE_BUILD and device is None:
        raise ValueError("An engine build requires its CUDA device for synchronization")
    try:
        target = None if device is None else _device(device)
        if target is not None:
            torch.cuda.synchronize(target)
        start = perf_counter_ns()
        operation()
        if target is not None:
            torch.cuda.synchronize(target)
        return ConversionResult(phase, _wall_ms(start))
    except (CalledProcessError, OSError, RuntimeError, ValueError) as error:
        return ConversionResult(
            phase, None, _failure(error, phase, Status.CONVERSION_FAILED)
        )


class _BenchmarkNvml(NvmlSampler):
    def memory_used_bytes(self, device: str) -> int:
        memory = _NvmlMemory()
        self._call(
            "nvmlDeviceGetMemoryInfo", self._handles[device], ctypes.byref(memory)
        )
        return int(memory.used)

    def device_state(self, device: str) -> dict[str, object]:
        """Read driver, clock and power telemetry once, preserving unavailable fields."""
        driver = ctypes.create_string_buffer(96)
        self._call("nvmlSystemGetDriverVersion", driver, len(driver))
        handle = self._handles[device]
        result: dict[str, object] = {"driver": driver.value.decode("ascii")}
        errors: dict[str, str] = {}
        for metric, function, args in (
            ("sm_clock_mhz", "nvmlDeviceGetClockInfo", (1,)),
            ("memory_clock_mhz", "nvmlDeviceGetClockInfo", (2,)),
            ("max_sm_clock_mhz", "nvmlDeviceGetMaxClockInfo", (1,)),
            ("max_memory_clock_mhz", "nvmlDeviceGetMaxClockInfo", (2,)),
            ("power_mw", "nvmlDeviceGetPowerUsage", ()),
            ("power_limit_mw", "nvmlDeviceGetEnforcedPowerLimit", ()),
        ):
            value = ctypes.c_uint()
            try:
                self._call(function, handle, *args, ctypes.byref(value))
            except (AttributeError, RuntimeError) as error:
                errors[metric] = str(error)
                result[metric] = None
            else:
                result[metric] = value.value
        result["telemetry_errors"] = errors
        return result


class _CudaMeasurements:
    def __init__(self, device: torch.device, nvml: _BenchmarkNvml) -> None:
        self.device = device
        self.nvml = nvml
        self.events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self.events[0].record()
        self.events[1].record()
        self.synchronize()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    def memory_used_bytes(self) -> int:
        return self.nvml.memory_used_bytes(str(self.device))

    def torch_peak_bytes(self) -> tuple[int, int]:
        return (
            torch.cuda.max_memory_allocated(self.device),
            torch.cuda.max_memory_reserved(self.device),
        )


@contextmanager
def _cuda_measurements(
    device: str | torch.device, environment: dict[str, object]
) -> Iterator[_CudaMeasurements]:
    target = _device(device)
    matmul, cudnn, conv = (
        torch.backends.cuda.matmul,
        torch.backends.cudnn,
        torch.backends.cudnn.conv,
    )
    initial = matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision
    with (
        torch.cuda.device(target),
        torch.no_grad(),
        torch.autocast("cuda", enabled=False),
    ):
        try:
            matmul.fp32_precision = cudnn.fp32_precision = conv.fp32_precision = "ieee"
            target = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(target)
            uuid = str(properties.uuid)
            try:
                nvml = _BenchmarkNvml({str(target): uuid})
            except (OSError, RuntimeError) as error:
                raise ConditionUnavailable(Status.UNAVAILABLE, str(error)) from error
            try:
                environment.update(
                    gpu=properties.name,
                    gpu_uuid=uuid,
                    gpu_total_bytes=properties.total_memory,
                    compute_capability=(properties.major, properties.minor),
                    cuda_device=str(target),
                    torch=str(torch.__version__),
                    cuda=torch.version.cuda,
                    tensorrt={
                        name: metadata.version(name)
                        for name in metadata.packages_distributions().get(
                            "tensorrt", ()
                        )
                    },
                    fp32_precision={
                        "matmul": matmul.fp32_precision,
                        "cudnn": cudnn.fp32_precision,
                        "cudnn_conv": conv.fp32_precision,
                    },
                    **nvml.device_state(str(target)),
                )
                cuda = _CudaMeasurements(target, nvml)
                torch.cuda.reset_peak_memory_stats(target)
                yield cuda
            finally:
                nvml.close()
        finally:
            matmul.fp32_precision, cudnn.fp32_precision, conv.fp32_precision = initial


class BatchStages:
    """Serial stage boundaries for one adapter invocation."""

    def __init__(
        self, cuda: _CudaMeasurements, backend: Backend, *, timed: bool
    ) -> None:
        self._cuda = cuda
        self._backend = backend
        self._timed = timed
        self._samples: list[StageSample] = []
        self._forward_count = 0
        self._error: Exception | None = None
        self._running = False

    def stage(
        self, stage: Stage, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        if stage is Stage.FORWARD:
            raise ValueError(
                "Use measure_forward() to declare the backend for model execution"
            )
        return self._measure(stage, name, operation)

    def measure_forward(
        self, backend: Backend, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        if self._error is not None:
            raise self._error
        if backend is not self._backend:
            self._error = ConditionUnavailable(
                Status.BACKEND_MISMATCH,
                f"Requested {self._backend}, forward declared {backend}",
            )
            raise self._error
        value = self._measure(Stage.FORWARD, name, operation)
        self._forward_count += 1
        return value

    def _measure(
        self, stage: Stage, name: str, operation: Callable[[], Value]
    ) -> Value:
        if self._error is not None:
            raise self._error
        if self._running:
            raise ValueError("Benchmark stages must run serially without nesting")
        timed_forward = self._timed and stage is Stage.FORWARD
        self._running = True
        try:
            start_event, end_event = self._cuda.events
            if timed_forward:
                start_event.record()
            start = perf_counter_ns()
            value = operation()
            if timed_forward:
                end_event.record()
            self._cuda.synchronize()
            wall_ms = _wall_ms(start)
            if self._timed:
                gpu_ms = start_event.elapsed_time(end_event) if timed_forward else None
                self._samples.append(StageSample(stage, name, wall_ms, gpu_ms))
            return value
        except Exception as error:
            self._error = error
            raise
        finally:
            self._running = False

    def _finish(self) -> tuple[StageSample, ...]:
        if self._error is not None:
            raise self._error
        if self._forward_count == 0:
            raise ValueError(
                "The adapter must execute a declared forward for every batch"
            )
        return tuple(self._samples)


def _percentile(ordered_ms: list[float], q: float) -> float:
    position = (len(ordered_ms) - 1) * q
    index = floor(position)
    if index == len(ordered_ms) - 1:
        return ordered_ms[index]
    return ordered_ms[index] + (position - index) * (
        ordered_ms[index + 1] - ordered_ms[index]
    )


def _stats(pairs: Sequence[tuple[float, int]]) -> TimingStats:
    durations = sorted(elapsed for elapsed, _ in pairs)
    if any(not isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Measured latencies must be finite and positive")
    per_item = sorted(elapsed / count for elapsed, count in pairs)
    items = sum(count for _, count in pairs)
    total = sum(durations)
    return TimingStats(
        samples=len(pairs),
        items=items,
        total_ms=total,
        p50_ms=_percentile(durations, 0.5),
        p95_ms=_percentile(durations, 0.95),
        p50_ms_per_item=_percentile(per_item, 0.5),
        p95_ms_per_item=_percentile(per_item, 0.95),
        items_per_second=1000 * items / total,
    )


def _summarize(samples: Sequence[BatchSample]) -> dict[str, TimingStats]:
    groups: dict[str, list[tuple[float, int]]] = {"stage_wall": [], "model_gpu": []}
    for sample in samples:
        groups["stage_wall"].append((sample.wall_ms, sample.items))
        totals: dict[str, float] = {}
        gpu_ms = 0.0
        for stage in sample.stages:
            totals[stage.stage] = totals.get(stage.stage, 0.0) + stage.wall_ms
            if stage.gpu_ms is not None:
                gpu_ms += stage.gpu_ms
        groups["model_gpu"].append((gpu_ms, sample.items))
        for stage_name, wall_ms in totals.items():
            groups.setdefault(stage_name, []).append((wall_ms, sample.items))
    return {name: _stats(pairs) for name, pairs in groups.items()}


def _fingerprint(
    files: Mapping[str, Path], destination: dict[str, dict[str, object]]
) -> None:
    for label, path in files.items():
        if not path.is_file():
            raise ConditionUnavailable(
                Status.UNAVAILABLE, f"Missing artifact {label}: {path}"
            )
        destination[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": dual_sha256(path),
        }


def benchmark(
    case: BenchmarkCase,
    batches: Sequence[Batch[Input]],
    backends: Mapping[Backend, BenchmarkAdapter[Input, Output]],
    *,
    device: str | torch.device = "cuda:0",
    protocol: BenchmarkProtocol = _DEFAULT_PROTOCOL,
    conversion: Sequence[ConversionResult] = (),
    check_parity: Callable[[Iterator[tuple[Batch[Input], Output]]], ParityResult]
    | None = None,
) -> BenchmarkReport:
    """Measure one backend and batch size on the supplied ordered, full batches."""
    batches = tuple(batches)
    if case.batch_size not in protocol.batch_sizes or not batches:
        raise ValueError("Provide pinned batches for a size in the benchmark sweep")
    if any(len(batch.ids) != case.batch_size for batch in batches):
        raise ValueError("Every batch must match the exact requested size")
    if len({result.phase for result in conversion}) != len(conversion):
        raise ValueError("Supply at most one result per conversion phase")
    report = BenchmarkReport(case, protocol, tuple(conversion))
    for result in conversion:
        if result.failure is not None:
            report.failure = result.failure
            return report

    phase = "backend"
    try:
        adapter = backends.get(case.backend)
        if adapter is None:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, f"No adapter registered for {case.backend}"
            )
        if adapter.backend is not case.backend:
            raise ConditionUnavailable(
                Status.BACKEND_MISMATCH,
                f"Requested {case.backend}, adapter declared {adapter.backend}",
            )
        phase = "assets"
        _fingerprint(adapter.weights, report.weights)
        _fingerprint(adapter.engines, report.engines)
        if case.backend is Backend.TRT16 and not report.engines:
            raise ConditionUnavailable(
                Status.UNAVAILABLE, "TRT16 requires an engine artifact"
            )
        phase = "environment"
        with _cuda_measurements(device, report.environment) as cuda:
            try:
                _run_case(report, cuda, batches, adapter, check_parity)
            finally:
                report.torch_peak_allocated_bytes, report.torch_peak_reserved_bytes = (
                    cuda.torch_peak_bytes()
                )
    except (ConditionUnavailable, torch.OutOfMemoryError) as error:
        report.failure = _failure(error, phase, Status.UNAVAILABLE)
    return report


def _run_case(
    report: BenchmarkReport,
    cuda: _CudaMeasurements,
    batches: Sequence[Batch[Input]],
    adapter: BenchmarkAdapter[Input, Output],
    check_parity: Callable[[Iterator[tuple[Batch[Input], Output]]], ParityResult]
    | None,
) -> None:
    def invoke(
        batch: Batch[Input], *, timed: bool = False
    ) -> tuple[Output, tuple[StageSample, ...]]:
        stages = BatchStages(cuda, report.case.backend, timed=timed)
        output = run(batch.data, stages)
        return output, stages._finish()

    phase = "load"
    try:
        report.nvml_used_bytes["before_load"] = cuda.memory_used_bytes()
        start = perf_counter_ns()
        run = adapter.load()
        cuda.synchronize()
        report.setup_wall_ms["load"] = _wall_ms(start)
        report.nvml_used_bytes["after_load"] = cuda.memory_used_bytes()

        phase = "cold_start"
        start = perf_counter_ns()
        invoke(batches[0])
        report.setup_wall_ms["cold_start"] = _wall_ms(start)
        report.nvml_used_bytes["after_cold_start"] = cuda.memory_used_bytes()

        phase = "warmup"
        inputs = cycle(batches)
        for _ in range(report.protocol.warmup):
            invoke(next(inputs))
        report.nvml_used_bytes["after_warmup"] = cuda.memory_used_bytes()

        phase = "measure"
        inputs = cycle(batches)
        for block in range(1, report.protocol.blocks + 1):
            for _ in range(report.protocol.iterations_per_block):
                batch = next(inputs)
                start = perf_counter_ns()
                output, stages = invoke(batch, timed=True)
                elapsed_ms = _wall_ms(start)
                del output
                report.samples.append(
                    BatchSample(
                        block,
                        batch.ids,
                        batch.item_count,
                        batch.processed_items,
                        elapsed_ms,
                        stages,
                    )
                )
            report.nvml_used_bytes[f"block_{block}"] = cuda.memory_used_bytes()

        phase = "parity"
        if check_parity is not None:
            observed = 0

            def predictions() -> Iterator[tuple[Batch[Input], Output]]:
                nonlocal observed
                for batch in batches:
                    output, _ = invoke(batch)
                    observed += 1
                    yield batch, output

            report.parity = check_parity(predictions())
            report.nvml_used_bytes["after_parity"] = cuda.memory_used_bytes()
            if not report.parity.passed:
                raise ConditionUnavailable(
                    Status.PARITY_FAILED,
                    report.parity.reason or "Output parity limits were exceeded",
                )
            if observed != len(batches):
                raise ValueError(
                    "A successful parity check must examine every supplied batch"
                )
        elif report.case.backend is not Backend.PT32:
            raise ConditionUnavailable(
                Status.PARITY_UNAVAILABLE, "A PT32 reference check is required"
            )
        report.summary = _summarize(report.samples)
    except (ConditionUnavailable, torch.OutOfMemoryError) as error:
        report.failure = _failure(error, phase, Status.UNAVAILABLE)
