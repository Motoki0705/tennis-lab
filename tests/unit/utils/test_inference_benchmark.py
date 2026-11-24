from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from src.utils import inference_benchmark as speed
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BatchStages,
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkProtocol,
    ConditionUnavailable,
    Conversion,
    ParityResult,
    Stage,
    Status,
    benchmark,
    measure_conversion,
)

SHORT = BenchmarkProtocol(
    batch_sizes=(1, 2), warmup=1, iterations_per_block=1, blocks=2
)


class _Event:
    def __init__(self, cuda: _Cuda) -> None:
        self.cuda = cuda
        self.timestamp_ms = 0.0

    def record(self) -> None:
        self.timestamp_ms = self.cuda.gpu_ms
        self.cuda.event_count += 1

    def elapsed_time(self, end: _Event) -> float:
        return end.timestamp_ms - self.timestamp_ms


class _Cuda:
    def __init__(self) -> None:
        self.host_ns = 0
        self.gpu_ms = 0.0
        self.pending_ms = 0
        self.event_count = 0
        self.events = _Event(self), _Event(self)
        self.used = iter((400, 520, 570, 580, 550, 630, 610))

    def step(self, wall_ms: int, gpu_ms: int = 0) -> None:
        self.host_ns += wall_ms * 1_000_000
        self.gpu_ms += gpu_ms

    def synchronize(self) -> None:
        self.step(1 + self.pending_ms, self.pending_ms)
        self.pending_ms = 0

    def memory_used_bytes(self) -> int:
        return next(self.used)

    def torch_peak_bytes(self) -> tuple[int, int]:
        return 180, 240


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch) -> _Cuda:
    runtime = _Cuda()

    @contextmanager
    def open_runtime(
        device: str | torch.device,
        environment: dict[str, object],
        *,
        dedicated_stream: bool,
    ) -> Iterator[_Cuda]:
        assert str(device) == "cuda:0"
        environment["gpu"] = "test device"
        yield runtime

    monkeypatch.setattr(speed, "_cuda_measurements", open_runtime)
    monkeypatch.setattr(speed, "perf_counter_ns", lambda: runtime.host_ns)
    return runtime


def _case(backend: Backend, batch_size: int = 1) -> BenchmarkCase:
    return BenchmarkCase(
        "person pipeline", backend, batch_size, "person-frame", "a" * 64
    )


def _unexpected_load() -> Callable[[int, BatchStages], int]:
    raise AssertionError("Unexpected backend load")


def test_nvml_memory_and_clock_snapshots_allow_unavailable_power_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = object.__new__(speed._BenchmarkNvml)
    handle = ctypes.c_void_p(42)
    sampler._handles = {"cuda:0": handle}

    def call(name: str, *args: Any) -> None:
        if name == "nvmlSystemGetDriverVersion":
            args[0].value = b"580.82"
            return
        assert args[0] is handle
        if name == "nvmlDeviceGetMemoryInfo":
            args[1]._obj.used = 3 * 2**30
        elif name == "nvmlDeviceGetClockInfo":
            args[2]._obj.value = 1200 * args[1]
        elif name == "nvmlDeviceGetMaxClockInfo":
            args[2]._obj.value = 1600 * args[1]
        elif name == "nvmlDeviceGetPowerUsage":
            args[1]._obj.value = 50_000
        elif name == "nvmlDeviceGetEnforcedPowerLimit":
            raise RuntimeError("Power limit is unavailable")
        else:
            raise AssertionError(f"Unexpected NVML call: {name}")

    monkeypatch.setattr(sampler, "_call", call)
    assert sampler.memory_used_bytes("cuda:0") == 3 * 2**30
    assert sampler.device_state("cuda:0") == {
        "driver": "580.82",
        "sm_clock_mhz": 1200,
        "memory_clock_mhz": 2400,
        "max_sm_clock_mhz": 1600,
        "max_memory_clock_mhz": 3200,
        "power_mw": 50_000,
        "power_limit_mw": None,
        "telemetry_errors": {"power_limit_mw": "Power limit is unavailable"},
    }


def test_composite_stage_and_gpu_times_use_full_batch_counts(
    cuda: _Cuda, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"abc")
    calls: list[int] = []

    def infer(value: int, stages: BatchStages) -> int:
        calls.append(value)
        stages.stage(Stage.PREPROCESS, lambda: cuda.step(1))
        stages.stage(Stage.H2D, lambda: cuda.step(2))
        stages.measure_forward(
            Backend.PT32, lambda: cuda.step(value + 1, value), name="detector"
        )
        stages.stage(Stage.POSTPROCESS, lambda: cuda.step(2), name="boxes")
        stages.measure_forward(
            Backend.PT32, lambda: cuda.step(2 * value + 1, 2 * value), name="pose"
        )
        stages.stage(Stage.D2H, lambda: cuda.step(1))
        return value

    def load() -> Callable[[int, BatchStages], int]:
        cuda.step(7)
        return infer

    def capture(predictions: Iterator[tuple[Batch[int], int]]) -> ParityResult:
        assert [(batch.ids, output) for batch, output in predictions] == [
            (("a", "b"), 1),
            (("c", "d"), 3),
        ]
        return ParityResult(True, {"reference_batches": 2})

    report = benchmark(
        _case(Backend.PT32, batch_size=2),
        [
            Batch(("a", "b"), 1, items=2, processed_items=3),
            Batch(("c", "d"), 3, items=3, processed_items=5),
        ],
        {Backend.PT32: BenchmarkAdapter(Backend.PT32, load, {"model": checkpoint})},
        protocol=SHORT,
        check_parity=capture,
    )

    assert calls == [1, 1, 1, 3, 1, 3]  # cold, warm-up, two blocks, then parity
    assert cuda.event_count == 8  # one pair per measured forward
    assert report.status is Status.OK
    assert report.setup_wall_ms == {"load": 8, "cold_start": 18}
    assert report.summary is not None
    assert [sample.block for sample in report.samples] == [1, 2]
    assert [sample.processed_items for sample in report.samples] == [3, 5]
    assert report.summary["stage_wall"].p50_ms == 21
    assert report.summary["stage_wall"].p95_ms == pytest.approx(23.7)
    assert report.summary["stage_wall"].items_per_second == pytest.approx(5000 / 42)
    assert report.summary["model_gpu"].p50_ms == 6
    assert report.summary["model_gpu"].p95_ms == pytest.approx(8.7)
    assert report.summary["model_gpu"].p95_ms_per_item == pytest.approx(2.925)
    assert report.summary["model_gpu"].items_per_second == pytest.approx(1000 * 5 / 12)
    assert report.summary["forward"].p50_ms == 10
    assert [
        stage.name for stage in report.samples[0].stages if stage.gpu_ms is not None
    ] == ["detector", "pose"]
    assert (
        report.weights["model"]["sha256"]
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    result = report.to_dict()
    assert result["nvml_device_peak_sampled_bytes"] == 630
    assert result["nvml_peak_minus_before_load_bytes"] == 230
    assert result["torch_peak_allocated_bytes"] == 180
    assert result["torch_peak_reserved_bytes"] == 240


def test_batch_end_waits_for_unstaged_cuda_work_in_same_sample(cuda: _Cuda) -> None:
    def infer(tail_ms: int, stages: BatchStages) -> int:
        assert cuda.pending_ms == 0
        stages.measure_forward(Backend.PT32, lambda: cuda.step(1, 1))
        cuda.pending_ms = tail_ms
        return tail_ms

    report = benchmark(
        _case(Backend.PT32),
        [Batch(("short",), 10), Batch(("long",), 30)],
        {Backend.PT32: BenchmarkAdapter(Backend.PT32, lambda: infer, {})},
        protocol=SHORT,
    )

    assert cuda.pending_ms == 0
    assert report.setup_wall_ms["cold_start"] == 13
    assert [(sample.ids, sample.wall_ms) for sample in report.samples] == [
        (("short",), 13),
        (("long",), 33),
    ]
    assert report.summary is not None
    assert report.summary["model_gpu"].p95_ms == 1
    assert report.summary["stage_wall"].items_per_second == pytest.approx(2000 / 46)


@pytest.mark.parametrize("phase", ["load", "measure"])
def test_allocator_oom_stops_condition_and_keeps_completed_samples(
    cuda: _Cuda, phase: str
) -> None:
    calls = 0

    def infer(value: int, stages: BatchStages) -> int:
        nonlocal calls
        calls += 1

        def forward() -> int:
            if calls == 4:
                raise torch.OutOfMemoryError("device allocation failed")
            cuda.step(2, 1)
            return value

        return stages.measure_forward(Backend.PT32, forward)

    def load() -> Callable[[int, BatchStages], int]:
        if phase == "load":
            raise torch.OutOfMemoryError("weights allocation failed")
        return infer

    report = benchmark(
        _case(Backend.PT32),
        [Batch(("a",), 1)],
        {
            Backend.PT32: BenchmarkAdapter(Backend.PT32, load, {}),
            Backend.PT16: BenchmarkAdapter(Backend.PT16, _unexpected_load, {}),
        },
        protocol=SHORT,
    )
    assert report.failure is not None
    assert report.status is Status.INFEASIBLE
    assert report.failure.status is Status.OOM
    assert report.failure.phase == phase
    assert "allocation failed" in report.failure.message
    assert report.summary is None
    assert len(report.samples) == int(phase == "measure")
    assert report.torch_peak_reserved_bytes == 240


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            ConditionUnavailable(Status.UNSUPPORTED, "No converter for grid_sampler"),
            Status.UNSUPPORTED,
        ),
        (RuntimeError("ONNX conversion failed"), Status.CONVERSION_FAILED),
        (torch.OutOfMemoryError("export allocation failed"), Status.OOM),
    ],
)
def test_conversion_failure_never_reaches_a_backend(
    error: RuntimeError, status: Status
) -> None:
    def export() -> None:
        raise error

    conversion = measure_conversion(Conversion.ONNX_EXPORT, export)
    report = benchmark(
        _case(Backend.TRT16),
        [Batch(("a",), 1)],
        {
            Backend.TRT16: BenchmarkAdapter(Backend.TRT16, _unexpected_load, {}),
            Backend.PT32: BenchmarkAdapter(Backend.PT32, _unexpected_load, {}),
        },
        protocol=SHORT,
        conversion=[conversion],
    )
    assert report.status is Status.INFEASIBLE
    assert report.failure is not None
    assert report.failure.status is status
    assert report.failure.phase == "onnx_export"
    assert str(error) in report.failure.message
    assert report.summary is None
    assert conversion.wall_ms is None
    assert not (report.samples or report.setup_wall_ms)


@pytest.mark.parametrize("wrong_registration", [False, True])
def test_exact_backend_selection_never_loads_another_backend(
    wrong_registration: bool,
) -> None:
    pt = BenchmarkAdapter(Backend.PT32, _unexpected_load, {})
    backends = {Backend.PT32: pt}
    if wrong_registration:
        backends[Backend.TRT16] = pt
    report = benchmark(
        _case(Backend.TRT16), [Batch(("a",), 1)], backends, protocol=SHORT
    )
    assert report.status is Status.INFEASIBLE
    assert report.summary is None
    assert report.failure is not None and report.failure.phase == "backend"
    assert report.failure.status is (
        Status.BACKEND_MISMATCH if wrong_registration else Status.UNAVAILABLE
    )


def test_forward_cannot_recover_a_failed_trt_call_with_pt(
    cuda: _Cuda, tmp_path: Path
) -> None:
    engine = tmp_path / "engine.plan"
    engine.write_bytes(b"engine")

    def infer(value: int, stages: BatchStages) -> int:
        def trt() -> int:
            raise ConditionUnavailable(
                Status.UNSUPPORTED, "engine does not support input"
            )

        def pytorch() -> int:
            raise AssertionError("PyTorch fallback must be rejected before execution")

        with pytest.raises(ConditionUnavailable):
            stages.measure_forward(Backend.TRT16, trt)
        with pytest.raises(ConditionUnavailable):
            stages.measure_forward(Backend.PT32, pytorch)
        return value

    adapter = BenchmarkAdapter(Backend.TRT16, lambda: infer, {}, {"model": engine})
    report = benchmark(
        _case(Backend.TRT16),
        [Batch(("a",), 1)],
        {Backend.TRT16: adapter},
        protocol=SHORT,
    )
    assert report.status is Status.INFEASIBLE
    assert report.failure is not None and report.failure.phase == "cold_start"
    assert report.failure.status is Status.UNSUPPORTED
    assert report.summary is None and not report.samples


@pytest.mark.parametrize("has_reference", [False, True])
def test_parity_failure_retains_completed_speed_summary(
    cuda: _Cuda, has_reference: bool
) -> None:
    def infer(value: int, stages: BatchStages) -> int:
        return stages.measure_forward(Backend.PT16, lambda: forward(value))

    def forward(value: int) -> int:
        cuda.step(2, 1)
        return value

    def check(predictions: Iterator[tuple[Batch[int], int]]) -> ParityResult:
        _, difference = next(predictions)
        return ParityResult(False, {"error_px": difference}, "P95 exceeds 1 px")

    report = benchmark(
        _case(Backend.PT16),
        [Batch(("a",), 5)],
        {Backend.PT16: BenchmarkAdapter(Backend.PT16, lambda: infer, {})},
        protocol=SHORT,
        check_parity=check if has_reference else None,
    )
    assert report.status is Status.PARITY_FAILED
    assert report.failure is not None and report.failure.phase == "parity"
    assert report.failure.status is (
        Status.PARITY_FAILED if has_reference else Status.PARITY_UNAVAILABLE
    )
    assert report.summary is not None and len(report.samples) == SHORT.blocks
    assert report.summary["model_gpu"].p50_ms == 1
    if has_reference:
        assert report.parity is not None and report.parity.metrics == {"error_px": 5}


def test_sweep_requires_batch_one_and_rejects_smaller_batches() -> None:
    with pytest.raises(ValueError, match="start with batch 1"):
        BenchmarkProtocol(batch_sizes=(2, 4))
    with pytest.raises(ValueError, match="exact requested size"):
        benchmark(
            _case(Backend.PT32, batch_size=2),
            [Batch(("one item",), 1)],
            {},
            protocol=SHORT,
        )


def test_build_time_is_separate_from_trt_load_and_inference(
    cuda: _Cuda, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(speed, "_device", torch.device)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: cuda.synchronize())
    engine = tmp_path / "engine.plan"

    def build() -> None:
        engine.write_bytes(b"abc")
        cuda.step(11)

    conversion = measure_conversion(Conversion.ENGINE_BUILD, build, device="cuda:0")

    def infer(value: int, stages: BatchStages) -> int:
        def forward() -> int:
            cuda.step(4, 2)
            return value + 1

        return stages.measure_forward(Backend.TRT16, forward)

    def check(predictions: Iterator[tuple[Batch[int], int]]) -> ParityResult:
        result = {batch.ids[0]: value for batch, value in predictions}
        return ParityResult(result == {"a": 2}, {"predictions": result})

    report = benchmark(
        _case(Backend.TRT16),
        [Batch(("a",), 1)],
        {
            Backend.PT32: BenchmarkAdapter(Backend.PT32, _unexpected_load, {}),
            Backend.TRT16: BenchmarkAdapter(
                Backend.TRT16, lambda: infer, {}, {"model": engine}
            ),
        },
        protocol=SHORT,
        conversion=[conversion],
        check_parity=check,
    )
    assert report.status is Status.OK and report.summary is not None
    assert report.conversion[0].wall_ms == 12
    assert report.setup_wall_ms == {"load": 1, "cold_start": 6}
    assert report.summary["model_gpu"].p50_ms == 2
    assert (
        report.engines["model"]["sha256"]
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
