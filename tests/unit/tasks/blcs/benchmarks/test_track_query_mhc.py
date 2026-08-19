from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.tasks.blcs.benchmarks import track_query_mhc
from src.tasks.blcs.benchmarks.contracts import (
    ISSUE_NUMBER,
    BenchmarkContractError,
    build_source_record,
    utc_timestamp,
)


def _run(
    candidate: str,
    kind: str,
    latency_ms: float,
    memory_bytes: int,
    *,
    parity: str = "pass",
) -> dict[str, Any]:
    return {
        "case": "configured-training-lower-bound",
        "candidate": candidate,
        "kind": kind,
        "implementation": candidate,
        "available": True,
        "shape": track_query_mhc.PROFILE_SHAPE,
        "dtype": "float32",
        "warmup": track_query_mhc.WARMUP,
        "iterations": track_query_mhc.ITERATIONS,
        "latency": {"median_ms": latency_ms, "p95_ms": latency_ms * 1.1},
        "throughput": {"unit": "stream-sets/s", "value": 1000.0},
        "memory": {
            "peak_allocated_bytes": memory_bytes,
            "peak_reserved_bytes": memory_bytes,
        },
        "parity": {
            "status": parity,
            "forward_max_abs_error": 0.0,
            "forward_mean_abs_error": 0.0,
            "backward_max_abs_error": 0.0,
            "backward_mean_abs_error": 0.0,
            "atol": track_query_mhc.BACKWARD_ATOL,
            "rtol": track_query_mhc.BACKWARD_RTOL,
        },
        "unavailable_reason": None,
    }


def _document(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": "mhc",
        "generated_at_utc": utc_timestamp(),
        "source": build_source_record(
            track_query_mhc.repository_root(), track_query_mhc.SOURCE_FILES
        ),
        "environment": {
            "python": "3.11.0",
            "torch": "2.0.0",
            "cuda": "12.0",
            "gpu": "test-gpu",
            "compute_capability": "8.0",
        },
        "protocol": track_query_mhc.PROTOCOL,
        "runs": runs,
        "decision": track_query_mhc.decide_mhc_profile(runs),
        "risks": ["test evidence"],
    }


def test_go_requires_speed_memory_and_parity() -> None:
    runs = [
        _run("eager", "pytorch_reference", 12.0, 100),
        _run("compiled", "pytorch_reference", 10.0, 90),
        _run("custom_cuda_prototype", "cuda_prototype", 8.0, 90),
    ]
    decision = track_query_mhc.decide_mhc_profile(runs)

    assert decision["status"] == "GO"
    assert decision["best_reference"] == "compiled"
    assert decision["speedup"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("latency", "memory", "parity", "reason_fragment"),
    [
        (9.5, 90, "pass", "speedup"),
        (8.0, 91, "pass", "memory increased"),
        (8.0, 90, "fail", "parity failed"),
    ],
)
def test_no_go_identifies_each_failed_gate(
    latency: float, memory: int, parity: str, reason_fragment: str
) -> None:
    runs = [
        _run("eager", "pytorch_reference", 12.0, 100),
        _run("compiled", "pytorch_reference", 10.0, 90),
        _run(
            "custom_cuda_prototype",
            "cuda_prototype",
            latency,
            memory,
            parity=parity,
        ),
    ]

    decision = track_query_mhc.decide_mhc_profile(runs)

    assert decision["status"] == "NO-GO"
    assert reason_fragment in decision["reason"]


def test_validation_rejects_tampered_decision() -> None:
    runs = [
        _run("eager", "pytorch_reference", 12.0, 100),
        _run("compiled", "pytorch_reference", 10.0, 90),
        _run("custom_cuda_prototype", "cuda_prototype", 9.5, 90),
    ]
    document = _document(runs)
    document["decision"]["status"] = "GO"

    with pytest.raises(BenchmarkContractError, match="decision"):
        track_query_mhc.validate_mhc_evidence(document)


def test_canonical_execute_does_not_modify_stable_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = [
        _run("eager", "pytorch_reference", 12.0, 100),
        _run("compiled", "pytorch_reference", 10.0, 90),
        _run("custom_cuda_prototype", "cuda_prototype", 9.5, 90),
    ]
    stable = _document(runs)
    runtime = copy.deepcopy(stable)
    runtime["generated_at_utc"] = "2026-08-19T00:00:00Z"
    runtime["runs"][0]["latency"]["median_ms"] = 12.5
    evidence = tmp_path / "mhc.json"
    runtime_result = tmp_path / "runtime.json"
    track_query_mhc.write_json_atomic(evidence, stable)
    before = evidence.read_bytes()
    monkeypatch.setattr(track_query_mhc, "profile_mhc", lambda: runtime)

    result = track_query_mhc.execute(
        evidence_path=evidence,
        runtime_result_path=runtime_result,
        record_evidence=False,
    )

    assert result == runtime
    assert evidence.read_bytes() == before
    assert runtime_result.is_file()


def test_execute_requires_distinct_runtime_path(tmp_path: Path) -> None:
    path = tmp_path / "mhc.json"
    with pytest.raises(BenchmarkContractError, match="must differ"):
        track_query_mhc.execute(
            evidence_path=path,
            runtime_result_path=path,
            record_evidence=False,
        )
