"""Contract tests for the Issue #753 CSWA CUDA benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from src.tasks.blcs.benchmarks.contracts import BenchmarkContractError
from src.tasks.blcs.benchmarks.track_query_cswa import (
    DTYPES,
    ITERATIONS,
    MEASUREMENTS,
    REQUIRED_SPEEDUP,
    WARMUP,
    _run_case_name,
    _run_shape,
    decide_cswa_profile,
)


def _run(
    *,
    dtype_name: str,
    measurement: str,
    candidate: str,
    latency_ms: float,
    memory_bytes: int,
    parity: str = "pass",
    available: bool = True,
) -> dict[str, Any]:
    return {
        "case": _run_case_name(dtype_name, measurement),
        "candidate": candidate,
        "kind": "cuda_production" if candidate == "cuda" else "pytorch_reference",
        "implementation": "test-double",
        "available": available,
        "shape": _run_shape(measurement),
        "dtype": dtype_name,
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": (
            {"median_ms": latency_ms, "p95_ms": latency_ms} if available else None
        ),
        "throughput": ({"unit": "query-tokens/s", "value": 1.0} if available else None),
        "memory": (
            {
                "peak_allocated_bytes": memory_bytes,
                "peak_reserved_bytes": memory_bytes,
            }
            if available
            else None
        ),
        "parity": {
            "status": parity if available else "not-run",
            "forward_max_abs_error": 0.0 if available else None,
            "forward_mean_abs_error": 0.0 if available else None,
            "backward_max_abs_error": 0.0 if available else None,
            "backward_mean_abs_error": 0.0 if available else None,
            "atol": 0.0 if available else None,
            "rtol": 0.0 if available else None,
        },
        "unavailable_reason": None if available else "test unavailable",
    }


def _passing_runs() -> list[dict[str, Any]]:
    return [
        _run(
            dtype_name=dtype_name,
            measurement=measurement,
            candidate=candidate,
            latency_ms=12.0 if candidate == "reference" else 9.0,
            memory_bytes=120 if candidate == "reference" else 80,
        )
        for dtype_name in DTYPES
        for measurement in MEASUREMENTS
        for candidate in ("reference", "cuda")
    ]


def test_decision_requires_both_dtypes_and_measurements_to_pass_all_gates() -> None:
    decision = decide_cswa_profile(_passing_runs())

    assert decision["status"] == "GO"
    assert decision["optimized_candidate"] == "cuda"
    cases = cast(Mapping[str, Mapping[str, Any]], decision["cases"])
    assert len(cases) == len(DTYPES) * len(MEASUREMENTS)
    assert all(case["speedup"] >= REQUIRED_SPEEDUP for case in cases.values())
    assert all(case["memory_reduced"] is True for case in cases.values())
    assert all(case["parity_passed"] is True for case in cases.values())


@pytest.mark.parametrize("failed_gate", ["speed", "memory", "parity", "available"])
def test_decision_is_no_go_when_any_long_context_gate_fails(failed_gate: str) -> None:
    runs = _passing_runs()
    target = next(
        run for run in runs if run["candidate"] == "cuda" and run["dtype"] == "bfloat16"
    )
    if failed_gate == "speed":
        cast(dict[str, float], target["latency"])["median_ms"] = 11.0
    elif failed_gate == "memory":
        cast(dict[str, int], target["memory"])["peak_allocated_bytes"] = 120
    elif failed_gate == "parity":
        cast(dict[str, Any], target["parity"])["status"] = "fail"
    else:
        target["available"] = False

    decision = decide_cswa_profile(runs)

    assert decision["status"] == "NO-GO"
    assert decision["optimized_candidate"] is None


def test_decision_rejects_missing_reference_authority() -> None:
    runs = [run for run in _passing_runs() if run["candidate"] == "cuda"]

    decision = decide_cswa_profile(runs)

    assert decision["status"] == "NO-GO"
    assert "unavailable" in decision["reason"]


def test_benchmark_contract_error_remains_a_value_error() -> None:
    assert issubclass(BenchmarkContractError, ValueError)
