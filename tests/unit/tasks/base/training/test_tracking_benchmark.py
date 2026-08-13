"""Unit coverage for the shared tracking fusion benchmark contract."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from src.tasks.base.training.tracking_benchmark import (
    TrackingFusionBenchmarkResult,
    benchmark_tracking_fusion_cuda,
)


def _inputs(*, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "court_peak_uv": torch.empty(1, 3, 64, 7, 4, 2, device=device),
        "court_peak_score": torch.empty(1, 3, 64, 7, 4, device=device),
        "court_peak_covariance": torch.empty(
            1, 3, 64, 7, 4, 2, 2, device=device
        ),
        "court_peak_valid": torch.empty(
            1, 3, 64, 7, 4, device=device, dtype=torch.bool
        ),
    }


def test_result_emits_stable_evidence_keys() -> None:
    assert TrackingFusionBenchmarkResult(1.25, 3.5).as_metrics() == {
        "court_peak_fusion_latency_ms": 1.25,
        "court_peak_fusion_peak_memory_mb": 3.5,
    }


def test_benchmark_rejects_cpu_and_noncanonical_shape() -> None:
    inputs = _inputs(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        benchmark_tracking_fusion_cuda(lambda: None, inputs=inputs)
    inputs["court_peak_uv"] = torch.empty(1, 3, 64, 7, 2, 2)
    with pytest.raises(ValueError, match="benchmark shape"):
        benchmark_tracking_fusion_cuda(lambda: None, inputs=inputs)


def test_benchmark_requires_fixed_warmups_and_repeats() -> None:
    inputs = _inputs(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="exactly 10 warmups"):
        benchmark_tracking_fusion_cuda(lambda: None, inputs=inputs, warmups=9)
    with pytest.raises(ValueError, match="exactly 50 repeats"):
        benchmark_tracking_fusion_cuda(lambda: None, inputs=inputs, repeats=49)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_benchmark_synchronizes_and_measures_actual_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    inputs = _inputs(device=device)
    fusion_call = Mock(return_value=torch.empty(1, device=device))
    synchronize = Mock(wraps=torch.cuda.synchronize)
    reset_peak = Mock(wraps=torch.cuda.reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)

    result = benchmark_tracking_fusion_cuda(fusion_call, inputs=inputs)

    assert fusion_call.call_count == 60
    assert synchronize.call_count == 52
    assert all(
        args.args[0] == inputs["court_peak_uv"].device
        for args in synchronize.call_args_list
    )
    reset_peak.assert_called_once_with(inputs["court_peak_uv"].device)
    assert result.latency_ms >= 0.0
    assert result.peak_memory_mb >= 0.0
