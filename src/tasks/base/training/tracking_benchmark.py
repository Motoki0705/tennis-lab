"""CUDA benchmark contract for shared tracking observation fusion."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

TRACKING_FUSION_BENCHMARK_BATCH_SIZE = 1
TRACKING_FUSION_BENCHMARK_VIEWS = 3
TRACKING_FUSION_BENCHMARK_FRAMES = 64
TRACKING_FUSION_BENCHMARK_DETECTIONS = 4
TRACKING_FUSION_BENCHMARK_CLASSES = 7
TRACKING_FUSION_BENCHMARK_PEAKS = 4
TRACKING_FUSION_BENCHMARK_WARMUPS = 10
TRACKING_FUSION_BENCHMARK_REPEATS = 50


@dataclass(frozen=True, slots=True)
class TrackingFusionBenchmarkResult:
    """Measured latency and incremental peak allocation for one fusion path."""

    latency_ms: float
    peak_memory_mb: float

    def as_metrics(
        self, *, prefix: str = "court_peak_fusion"
    ) -> dict[str, float]:
        """Return stable metric keys consumed by experiment evidence checks."""
        if not prefix or prefix.endswith("_"):
            raise ValueError("benchmark metric prefix must be non-empty without '_'.")
        return {
            f"{prefix}_latency_ms": self.latency_ms,
            f"{prefix}_peak_memory_mb": self.peak_memory_mb,
        }


def _validate_fusion_inputs(inputs: Mapping[str, Tensor]) -> torch.device:
    expected_prefix = (
        TRACKING_FUSION_BENCHMARK_BATCH_SIZE,
        TRACKING_FUSION_BENCHMARK_VIEWS,
        TRACKING_FUSION_BENCHMARK_FRAMES,
    )
    required_shapes = {
        "court_peak_uv": (
            *expected_prefix,
            TRACKING_FUSION_BENCHMARK_CLASSES,
            TRACKING_FUSION_BENCHMARK_PEAKS,
            2,
        ),
        "court_peak_score": (
            *expected_prefix,
            TRACKING_FUSION_BENCHMARK_CLASSES,
            TRACKING_FUSION_BENCHMARK_PEAKS,
        ),
        "court_peak_covariance": (
            *expected_prefix,
            TRACKING_FUSION_BENCHMARK_CLASSES,
            TRACKING_FUSION_BENCHMARK_PEAKS,
            2,
            2,
        ),
        "court_peak_valid": (
            *expected_prefix,
            TRACKING_FUSION_BENCHMARK_CLASSES,
            TRACKING_FUSION_BENCHMARK_PEAKS,
        ),
    }
    missing = required_shapes.keys() - inputs.keys()
    if missing:
        raise ValueError(f"benchmark inputs are missing {sorted(missing)!r}.")
    for name, shape in required_shapes.items():
        if inputs[name].shape != shape:
            raise ValueError(f"{name} must have benchmark shape {shape}.")
    devices = {inputs[name].device for name in required_shapes}
    if len(devices) != 1:
        raise ValueError("benchmark inputs must share one CUDA device.")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("court peak fusion benchmarking requires CUDA tensors.")
    return device


def benchmark_tracking_fusion_cuda(
    fusion_call: Callable[[], object],
    *,
    inputs: Mapping[str, Tensor],
    warmups: int = TRACKING_FUSION_BENCHMARK_WARMUPS,
    repeats: int = TRACKING_FUSION_BENCHMARK_REPEATS,
) -> TrackingFusionBenchmarkResult:
    """Measure an actual task fusion call under the fixed Issue #719 shape.

    The callable must execute the same observation-fusion module used by the task
    model. CUDA synchronization brackets warmup, allocation reset, and timing.
    Peak memory is reported as CUDA's absolute peak allocated memory for the
    benchmark process after the warmup reset.
    """
    if warmups != TRACKING_FUSION_BENCHMARK_WARMUPS:
        raise ValueError("court peak fusion benchmark requires exactly 10 warmups.")
    if repeats != TRACKING_FUSION_BENCHMARK_REPEATS:
        raise ValueError("court peak fusion benchmark requires exactly 50 repeats.")
    device = _validate_fusion_inputs(inputs)
    torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmups):
            fusion_call()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms = 0.0
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fusion_call()
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms += float(start.elapsed_time(end))
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return TrackingFusionBenchmarkResult(
        latency_ms=elapsed_ms / repeats,
        peak_memory_mb=peak_bytes / (1024.0**2),
    )


__all__ = [
    "TRACKING_FUSION_BENCHMARK_BATCH_SIZE",
    "TRACKING_FUSION_BENCHMARK_CLASSES",
    "TRACKING_FUSION_BENCHMARK_DETECTIONS",
    "TRACKING_FUSION_BENCHMARK_FRAMES",
    "TRACKING_FUSION_BENCHMARK_PEAKS",
    "TRACKING_FUSION_BENCHMARK_REPEATS",
    "TRACKING_FUSION_BENCHMARK_VIEWS",
    "TRACKING_FUSION_BENCHMARK_WARMUPS",
    "TrackingFusionBenchmarkResult",
    "benchmark_tracking_fusion_cuda",
]
