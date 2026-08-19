from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from src.tasks.blcs.benchmarks import track_query_compressor
from src.tasks.blcs.benchmarks.contracts import (
    ISSUE_NUMBER,
    BenchmarkContractError,
    build_source_record,
    utc_timestamp,
)


def _run(
    case_index: int,
    candidate: str,
    kind: str,
    latency_ms: float,
    memory_bytes: int,
    *,
    parity: str = "pass",
) -> dict[str, Any]:
    case = track_query_compressor.PROFILE_CASES[case_index]
    return {
        "case": case.name,
        "candidate": candidate,
        "kind": kind,
        "implementation": candidate,
        "available": True,
        "shape": case.shape,
        "dtype": "float32",
        "warmup": track_query_compressor.WARMUP,
        "iterations": track_query_compressor.ITERATIONS,
        "latency": {"median_ms": latency_ms, "p95_ms": latency_ms * 1.1},
        "throughput": {"unit": "input-tokens/s", "value": 1000.0},
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
            "atol": track_query_compressor.BACKWARD_ATOL,
            "rtol": track_query_compressor.BACKWARD_RTOL,
        },
        "unavailable_reason": None,
    }


def _runs(
    *,
    custom_latency: tuple[float, float] = (8.0, 16.0),
    custom_memory: tuple[int, int] = (90, 180),
    custom_parity: tuple[str, str] = ("pass", "pass"),
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for case_index in range(len(track_query_compressor.PROFILE_CASES)):
        scale = case_index + 1
        runs.extend(
            [
                _run(
                    case_index, "eager", "pytorch_reference", 12.0 * scale, 100 * scale
                ),
                _run(
                    case_index,
                    "compiled",
                    "pytorch_reference",
                    10.0 * scale,
                    90 * scale,
                ),
                _run(
                    case_index,
                    "custom_cuda_prototype",
                    "cuda_prototype",
                    custom_latency[case_index],
                    custom_memory[case_index],
                    parity=custom_parity[case_index],
                ),
            ]
        )
    return runs


def _document(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": "compressor",
        "generated_at_utc": utc_timestamp(),
        "source": build_source_record(
            track_query_compressor.repository_root(),
            track_query_compressor.SOURCE_FILES,
        ),
        "environment": {
            "python": "3.11.0",
            "torch": "2.0.0",
            "cuda": "12.0",
            "gpu": "test-gpu",
            "compute_capability": "8.0",
        },
        "protocol": track_query_compressor.PROTOCOL,
        "runs": runs,
        "decision": track_query_compressor.decide_compressor_profile(runs),
        "risks": ["test evidence"],
    }


def test_go_requires_every_target_shape_to_pass() -> None:
    decision = track_query_compressor.decide_compressor_profile(_runs())

    assert decision["status"] == "GO"
    for case in decision["cases"].values():
        assert case["best_reference"] == "compiled"
        assert case["speedup"] == pytest.approx(1.25)
        assert case["passed"] is True


@pytest.mark.parametrize(
    ("latencies", "memories", "parities", "reason_fragment"),
    [
        ((8.0, 19.0), (90, 180), ("pass", "pass"), "speedup"),
        ((8.0, 16.0), (90, 181), ("pass", "pass"), "memory increased"),
        ((8.0, 16.0), (90, 180), ("pass", "fail"), "parity failed"),
    ],
)
def test_no_go_if_one_target_fails_any_gate(
    latencies: tuple[float, float],
    memories: tuple[int, int],
    parities: tuple[str, str],
    reason_fragment: str,
) -> None:
    decision = track_query_compressor.decide_compressor_profile(
        _runs(
            custom_latency=latencies,
            custom_memory=memories,
            custom_parity=parities,
        )
    )

    assert decision["status"] == "NO-GO"
    assert decision["optimized_candidate"] is None
    assert reason_fragment in decision["reason"]


def test_protocol_freezes_required_mask_parity_matrix() -> None:
    assert track_query_compressor.PROTOCOL["parity_cases"] == [
        "first-block-short-sequence",
        "partial-tail",
        "all-invalid",
        "sparse-mask",
    ]
    assert len(track_query_compressor.PROTOCOL["target_shapes"]) >= 2


def test_validation_rejects_tampered_decision() -> None:
    document = _document(_runs())
    document["decision"]["status"] = "NO-GO"

    with pytest.raises(BenchmarkContractError, match="decision"):
        track_query_compressor.validate_compressor_evidence(document)


def test_canonical_execute_does_not_modify_stable_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stable = _document(_runs(custom_latency=(9.5, 19.0)))
    runtime = copy.deepcopy(stable)
    runtime["generated_at_utc"] = "2026-08-19T00:00:00Z"
    runtime["runs"][0]["latency"]["median_ms"] = 12.5
    evidence = tmp_path / "compressor.json"
    runtime_result = tmp_path / "runtime.json"
    track_query_compressor.write_json_atomic(evidence, stable)
    before = evidence.read_bytes()
    monkeypatch.setattr(track_query_compressor, "profile_compressor", lambda: runtime)

    result = track_query_compressor.execute(
        evidence_path=evidence,
        runtime_result_path=runtime_result,
        record_evidence=False,
    )

    assert result == runtime
    assert evidence.read_bytes() == before
    assert runtime_result.is_file()


def test_execute_requires_distinct_runtime_path(tmp_path: Path) -> None:
    path = tmp_path / "compressor.json"
    with pytest.raises(BenchmarkContractError, match="must differ"):
        track_query_compressor.execute(
            evidence_path=path,
            runtime_result_path=path,
            record_evidence=False,
        )
