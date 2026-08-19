"""Benchmark fused CUDA compressed-window attention against its reference."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from src.tasks.blcs.benchmarks.contracts import (
    ISSUE_NUMBER,
    BenchmarkContractError,
    build_cuda_environment,
    build_source_record,
    load_json_object,
    repository_root,
    unavailable_run,
    utc_timestamp,
    validate_common_evidence,
    write_json_atomic,
)
from src.utils.models.components.ops.compressed_time_local import (
    resolve_compressed_time_local_attention,
)
from src.utils.models.components.ops.compressed_time_local.reference import (
    reference_compressed_time_local_attention,
)

COMPONENT = "cswa"
SOURCE_FILES = (
    "src/tasks/blcs/benchmarks/contracts.py",
    "src/tasks/blcs/benchmarks/track_query_cswa/__init__.py",
    "src/utils/configuration/operations.py",
    "src/utils/models/components/ops/" + "build.py",
    "src/utils/models/components/ops/compressed_time_local/_autograd.py",
    "src/utils/models/components/ops/compressed_time_local/api.py",
    "src/utils/models/components/ops/compressed_time_local/bindings.cpp",
    "src/utils/models/components/ops/compressed_time_local/kernels.cu",
    "src/utils/models/components/ops/compressed_time_local/layout.py",
    "src/utils/models/components/ops/compressed_time_local/reference.py",
    "src/utils/models/components/ops/loader.py",
)
SEED = ISSUE_NUMBER
WARMUP = 2
ITERATIONS = 5
COMPRESSION_RATIO = 4
WINDOW_RADIUS = 4
MASK_DENSITY = 0.875
REQUIRED_SPEEDUP = 1.20
DTYPES = ("float32", "bfloat16")
MEASUREMENTS = ("forward", "forward-backward")
PARITY_CASES = ("random", "boundary", "sparse-mask", "all-invalid")
TOLERANCES: dict[str, dict[str, float]] = {
    "float32": {
        "forward_atol": 1.0e-5,
        "forward_rtol": 1.0e-4,
        "backward_atol": 2.0e-5,
        "backward_rtol": 2.0e-4,
    },
    "bfloat16": {
        "forward_atol": 1.6e-2,
        "forward_rtol": 1.6e-2,
        # The edge-case matrix measured a 0.0625 worst-case difference (one
        # bfloat16 quantization step at the observed gradient magnitude) and a
        # 0.0023 worst mean error due to reference-vs-atomic reduction order.
        "backward_atol": 6.5e-2,
        "backward_rtol": 2.0e-2,
    },
}


@dataclass(frozen=True, slots=True)
class _ProfileCase:
    name: str = "long-context-query-path"
    batch_size: int = 16
    heads: int = 4
    query_length: int = 2048
    head_dim: int = 64

    @property
    def key_length(self) -> int:
        return (self.query_length + COMPRESSION_RATIO - 1) // COMPRESSION_RATIO

    @property
    def shape(self) -> dict[str, int | float]:
        return {
            "batch_size": self.batch_size,
            "heads": self.heads,
            "query_length": self.query_length,
            "key_length": self.key_length,
            "head_dim": self.head_dim,
            "compression_ratio": COMPRESSION_RATIO,
            "window_radius": WINDOW_RADIUS,
            "window_width": 2 * WINDOW_RADIUS + 1,
            "mask_density": MASK_DENSITY,
        }


PROFILE_CASE = _ProfileCase()
PROTOCOL: dict[str, Any] = {
    "seed": SEED,
    "measurement": list(MEASUREMENTS),
    "synchronize_each_iteration": True,
    "warmup": WARMUP,
    "iterations": ITERATIONS,
    "target_shapes": [PROFILE_CASE.shape],
    "dtypes": list(DTYPES),
    "production_mixed_precision_dtype": "bfloat16",
    "mask_density": MASK_DENSITY,
    "parity_cases": list(PARITY_CASES),
    "prototype_scope": (
        "one-warp-per-query online softmax without [N,H,T,Wc,Dh] K/V gather"
    ),
    "gate": {
        "minimum_forward_speedup": REQUIRED_SPEEDUP,
        "minimum_forward_backward_speedup": REQUIRED_SPEEDUP,
        "peak_allocated_memory_reduction_each_run": True,
        "dtype_tolerances": TOLERANCES,
    },
}

_WORKER_PREFIX = "CSWA_WORKER_RESULT="
_Executor = Callable[..., Tensor]
_Measurement = Literal["forward", "forward-backward"]


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported benchmark dtype: {name}")


def _make_masks(
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    query_valid = (
        torch.rand(
            batch_size,
            query_length,
            device="cuda",
            generator=generator,
        )
        < MASK_DENSITY
    )
    key_valid = (
        torch.rand(
            batch_size,
            key_length,
            device="cuda",
            generator=generator,
        )
        < MASK_DENSITY
    )
    window_width = 2 * WINDOW_RADIUS + 1
    key_valid[:, ::window_width] = True
    key_valid[:, -1] = True
    query_valid[0] = True
    if batch_size > 1:
        query_valid[1] = False
        key_valid[1] = False
    return query_valid, key_valid


def _make_profile_inputs(
    dtype_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    case = PROFILE_CASE
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + (0 if dtype_name == "float32" else 1))
    shape = (case.batch_size, case.heads, case.query_length, case.head_dim)
    compressed_shape = (
        case.batch_size,
        case.heads,
        case.key_length,
        case.head_dim,
    )
    dtype = _torch_dtype(dtype_name)
    query = torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn(
        *compressed_shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(
        *compressed_shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    query_valid, key_valid = _make_masks(
        batch_size=case.batch_size,
        query_length=case.query_length,
        key_length=case.key_length,
        generator=generator,
    )
    return query, key, value, query_valid, key_valid


def _run_iteration(
    executor: _Executor,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    measurement: _Measurement,
) -> None:
    query, key, value, query_valid, key_valid = inputs
    if measurement == "forward":
        with torch.no_grad():
            executor(
                query,
                key,
                value,
                query_valid=query_valid,
                key_valid=key_valid,
                compression_ratio=COMPRESSION_RATIO,
                window_radius=WINDOW_RADIUS,
                dropout_p=0.0,
                training=False,
            )
        return
    query.grad = None
    key.grad = None
    value.grad = None
    output = executor(
        query,
        key,
        value,
        query_valid=query_valid,
        key_valid=key_valid,
        compression_ratio=COMPRESSION_RATIO,
        window_radius=WINDOW_RADIUS,
        dropout_p=0.0,
        training=False,
    )
    output.float().square().mean().backward()


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _measure_candidate(
    executor: _Executor,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
    measurement: _Measurement,
) -> tuple[dict[str, float], dict[str, int], dict[str, float | str]]:
    if measurement == "forward-backward":
        for tensor in inputs[:3]:
            tensor.requires_grad_(True)
    for _ in range(WARMUP):
        _run_iteration(executor, inputs, measurement)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _run_iteration(executor, inputs, measurement)
    torch.cuda.synchronize()
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    elapsed_ms: list[float] = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _run_iteration(executor, inputs, measurement)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    elapsed_ms.sort()
    median = _percentile(elapsed_ms, 0.5)
    p95 = _percentile(elapsed_ms, 0.95)
    tokens = PROFILE_CASE.batch_size * PROFILE_CASE.query_length
    return (
        {"median_ms": median, "p95_ms": p95},
        memory,
        {"unit": "query-tokens/s", "value": tokens / (median / 1000.0)},
    )


def _tensor_error(reference: Tensor, actual: Tensor) -> tuple[float, float]:
    difference = (reference.float() - actual.float()).abs()
    if difference.numel() == 0:
        return 0.0, 0.0
    return float(difference.max().item()), float(difference.mean().item())


def _parity_inputs(
    case_name: str,
    dtype_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int]:
    if case_name == "random":
        query_length, ratio, radius = 65, 4, 4
        head_dim = 64
        query_valid = None
        key_valid = None
    elif case_name == "boundary":
        query_length, ratio, radius = 17, 4, 2
        head_dim = 16
        query_valid = torch.ones(2, query_length, dtype=torch.bool, device="cuda")
        key_valid = torch.ones(
            2,
            (query_length + ratio - 1) // ratio,
            dtype=torch.bool,
            device="cuda",
        )
    elif case_name == "sparse-mask":
        query_length, ratio, radius = 33, 4, 2
        head_dim = 16
        query_valid = torch.tensor(
            [
                [(index % 3) != 1 for index in range(query_length)],
                [(index % 4) == 0 for index in range(query_length)],
            ],
            dtype=torch.bool,
            device="cuda",
        )
        key_valid = torch.tensor(
            [[True, False, True, False, True, False, True, False, True]] * 2,
            dtype=torch.bool,
            device="cuda",
        )
    elif case_name == "all-invalid":
        query_length, ratio, radius = 19, 4, 2
        head_dim = 16
        query_valid = torch.zeros(2, query_length, dtype=torch.bool, device="cuda")
        key_valid = torch.zeros(
            2,
            (query_length + ratio - 1) // ratio,
            dtype=torch.bool,
            device="cuda",
        )
    else:
        raise ValueError(f"unknown CSWA parity case: {case_name}")

    key_length = (query_length + ratio - 1) // ratio
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + query_length + (0 if dtype_name == "float32" else 1))
    dtype = _torch_dtype(dtype_name)
    # Slice a doubled feature dimension so the random case also covers the CUDA
    # boundary's explicit normalization of non-contiguous public inputs.
    query = torch.randn(
        2,
        3,
        query_length,
        2 * head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )[..., ::2]
    key = torch.randn(
        2,
        3,
        key_length,
        2 * head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )[..., ::2]
    value = torch.randn(
        2,
        3,
        key_length,
        2 * head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )[..., ::2]
    if query_valid is None or key_valid is None:
        query_valid, key_valid = _make_masks(
            batch_size=2,
            query_length=query_length,
            key_length=key_length,
            generator=generator,
        )
    return query, key, value, query_valid, key_valid, ratio, radius


def _parity_for_dtype(cuda_executor: _Executor, dtype_name: str) -> dict[str, Any]:
    tolerance = TOLERANCES[dtype_name]
    forward_errors: list[tuple[float, float]] = []
    backward_errors: list[tuple[float, float]] = []
    all_close = True
    for case_name in PARITY_CASES:
        inputs = _parity_inputs(case_name, dtype_name)
        reference_inputs = tuple(
            tensor.detach().clone().requires_grad_(index < 3)
            if isinstance(tensor, Tensor)
            else tensor
            for index, tensor in enumerate(inputs)
        )
        candidate_inputs = tuple(
            tensor.detach().clone().requires_grad_(index < 3)
            if isinstance(tensor, Tensor)
            else tensor
            for index, tensor in enumerate(inputs)
        )
        reference_query, reference_key, reference_value = cast(
            tuple[Tensor, Tensor, Tensor], reference_inputs[:3]
        )
        candidate_query, candidate_key, candidate_value = cast(
            tuple[Tensor, Tensor, Tensor], candidate_inputs[:3]
        )
        query_valid = inputs[3]
        key_valid = inputs[4]
        ratio = inputs[5]
        radius = inputs[6]
        reference_output = reference_compressed_time_local_attention(
            reference_query,
            reference_key,
            reference_value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=ratio,
            window_radius=radius,
        )
        candidate_output = cuda_executor(
            candidate_query,
            candidate_key,
            candidate_value,
            query_valid=query_valid,
            key_valid=key_valid,
            compression_ratio=ratio,
            window_radius=radius,
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(SEED + query_valid.shape[1])
        upstream = torch.randn(
            reference_output.shape,
            device="cuda",
            dtype=reference_output.dtype,
            generator=generator,
        )
        reference_output.backward(upstream)
        candidate_output.backward(upstream)

        forward_errors.append(_tensor_error(reference_output, candidate_output))
        all_close = all_close and torch.allclose(
            reference_output,
            candidate_output,
            atol=tolerance["forward_atol"],
            rtol=tolerance["forward_rtol"],
        )
        for reference_tensor, candidate_tensor in zip(
            (reference_query, reference_key, reference_value),
            (candidate_query, candidate_key, candidate_value),
            strict=True,
        ):
            if reference_tensor.grad is None or candidate_tensor.grad is None:
                raise RuntimeError("CSWA parity tensor did not receive a gradient")
            backward_errors.append(
                _tensor_error(reference_tensor.grad, candidate_tensor.grad)
            )
            all_close = all_close and torch.allclose(
                reference_tensor.grad,
                candidate_tensor.grad,
                atol=tolerance["backward_atol"],
                rtol=tolerance["backward_rtol"],
            )

    return {
        "status": "pass" if all_close else "fail",
        "forward_max_abs_error": max(error[0] for error in forward_errors),
        "forward_mean_abs_error": max(error[1] for error in forward_errors),
        "backward_max_abs_error": max(error[0] for error in backward_errors),
        "backward_mean_abs_error": max(error[1] for error in backward_errors),
        "atol": tolerance["backward_atol"],
        "rtol": tolerance["backward_rtol"],
    }


def _reference_self_parity(dtype_name: str) -> dict[str, Any]:
    tolerance = TOLERANCES[dtype_name]
    return {
        "status": "pass",
        "forward_max_abs_error": 0.0,
        "forward_mean_abs_error": 0.0,
        "backward_max_abs_error": 0.0,
        "backward_mean_abs_error": 0.0,
        "atol": tolerance["backward_atol"],
        "rtol": tolerance["backward_rtol"],
    }


def _run_case_name(dtype_name: str, measurement: str) -> str:
    return f"{PROFILE_CASE.name}-{dtype_name}-{measurement}"


def _run_shape(measurement: str) -> dict[str, int | float]:
    return {
        **PROFILE_CASE.shape,
        "includes_backward": int(measurement == "forward-backward"),
    }


def _available_run(
    *,
    candidate: str,
    dtype_name: str,
    measurement: str,
    latency: Mapping[str, float],
    memory: Mapping[str, int],
    throughput: Mapping[str, float | str],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    is_cuda = candidate == "cuda"
    return {
        "case": _run_case_name(dtype_name, measurement),
        "candidate": candidate,
        "kind": "cuda_production" if is_cuda else "pytorch_reference",
        "implementation": (
            "handwritten_cuda_online_softmax" if is_cuda else "gather_sdpa_reference"
        ),
        "available": True,
        "shape": _run_shape(measurement),
        "dtype": dtype_name,
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": dict(latency),
        "throughput": dict(throughput),
        "memory": dict(memory),
        "parity": dict(parity),
        "unavailable_reason": None,
    }


def _run_worker(candidate: str, dtype_name: str, measurement: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch CUDA is unavailable")
    if measurement not in MEASUREMENTS:
        raise ValueError(f"unsupported measurement: {measurement}")
    executor: _Executor = reference_compressed_time_local_attention
    parity = _reference_self_parity(dtype_name)
    if candidate == "cuda":
        executor = resolve_compressed_time_local_attention(
            "cuda",
            compression_ratio=COMPRESSION_RATIO,
            window_radius=WINDOW_RADIUS,
        )
        parity = _parity_for_dtype(executor, dtype_name)
    elif candidate != "reference":
        raise ValueError(f"unsupported worker candidate: {candidate}")
    inputs = _make_profile_inputs(dtype_name)
    latency, memory, throughput = _measure_candidate(
        executor,
        inputs,
        cast(_Measurement, measurement),
    )
    return _available_run(
        candidate=candidate,
        dtype_name=dtype_name,
        measurement=measurement,
        latency=latency,
        memory=memory,
        throughput=throughput,
        parity=parity,
    )


def _profile_subprocess(
    candidate: str,
    dtype_name: str,
    measurement: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.tasks.blcs.benchmarks.track_query_cswa",
        "--worker-candidate",
        candidate,
        "--worker-dtype",
        dtype_name,
        "--worker-measurement",
        measurement,
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        reason_lines = (completed.stderr or completed.stdout).strip().splitlines()
        reason = (
            reason_lines[-1]
            if reason_lines
            else f"worker exited {completed.returncode}"
        )
        return unavailable_run(
            case=_run_case_name(dtype_name, measurement),
            candidate=candidate,
            kind="cuda_production" if candidate == "cuda" else "pytorch_reference",
            implementation=(
                "handwritten_cuda_online_softmax"
                if candidate == "cuda"
                else "gather_sdpa_reference"
            ),
            shape=_run_shape(measurement),
            dtype=dtype_name,
            warmup=WARMUP,
            iterations=ITERATIONS,
            reason=reason[:1000],
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_PREFIX):
            payload = json.loads(line.removeprefix(_WORKER_PREFIX))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
            break
    raise RuntimeError(
        f"worker {candidate}/{dtype_name}/{measurement} did not emit a result"
    )


def decide_cswa_profile(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the 1.20x latency, reduced-memory, and dtype-parity gate."""
    cases: dict[str, dict[str, Any]] = {}
    failed_reasons: list[str] = []
    for dtype_name in DTYPES:
        for measurement in MEASUREMENTS:
            case_name = _run_case_name(dtype_name, measurement)
            case_runs = [run for run in runs if run.get("case") == case_name]
            reference = next(
                (run for run in case_runs if run.get("candidate") == "reference"),
                None,
            )
            optimized = next(
                (run for run in case_runs if run.get("candidate") == "cuda"),
                None,
            )
            if (
                reference is None
                or optimized is None
                or reference.get("available") is not True
                or optimized.get("available") is not True
            ):
                reason = "reference or CUDA production run unavailable"
                cases[case_name] = {
                    "speedup": None,
                    "memory_reduced": False,
                    "parity_passed": False,
                    "passed": False,
                }
                failed_reasons.append(f"{case_name}: {reason}")
                continue
            reference_latency = float(
                cast(Mapping[str, Any], reference["latency"])["median_ms"]
            )
            optimized_latency = float(
                cast(Mapping[str, Any], optimized["latency"])["median_ms"]
            )
            speedup = reference_latency / optimized_latency
            reference_memory = cast(Mapping[str, Any], reference["memory"])
            optimized_memory = cast(Mapping[str, Any], optimized["memory"])
            memory_reduced = int(optimized_memory["peak_allocated_bytes"]) < int(
                reference_memory["peak_allocated_bytes"]
            )
            parity_passed = (
                cast(Mapping[str, Any], optimized["parity"])["status"] == "pass"
            )
            passed = speedup >= REQUIRED_SPEEDUP and memory_reduced and parity_passed
            cases[case_name] = {
                "speedup": speedup,
                "memory_reduced": memory_reduced,
                "parity_passed": parity_passed,
                "passed": passed,
            }
            gates: list[str] = []
            if speedup < REQUIRED_SPEEDUP:
                gates.append(f"speedup {speedup:.3f}x < {REQUIRED_SPEEDUP:.2f}x")
            if not memory_reduced:
                gates.append("peak allocated memory was not reduced")
            if not parity_passed:
                gates.append("forward/backward parity failed")
            if gates:
                failed_reasons.append(f"{case_name}: {', '.join(gates)}")

    passed_all = all(case["passed"] for case in cases.values())
    return {
        "status": "GO" if passed_all else "NO-GO",
        "required_speedup": REQUIRED_SPEEDUP,
        "optimized_candidate": "cuda" if passed_all else None,
        "cases": cases,
        "reason": (
            "Fused compressed-window CUDA passed latency, memory, and dtype gates."
            if passed_all
            else "No production CUDA backend is registered: "
            + "; ".join(failed_reasons)
            + "."
        ),
    }


def profile_cswa() -> dict[str, Any]:
    """Run reference and fused CUDA in isolated subprocesses."""
    environment = build_cuda_environment()
    runs = [
        _profile_subprocess(candidate, dtype_name, measurement)
        for dtype_name in DTYPES
        for measurement in MEASUREMENTS
        for candidate in ("reference", "cuda")
    ]
    report = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": COMPONENT,
        "generated_at_utc": utc_timestamp(),
        "source": build_source_record(repository_root(), SOURCE_FILES),
        "environment": environment,
        "protocol": PROTOCOL,
        "runs": runs,
        "decision": decide_cswa_profile(runs),
        "risks": [
            "K/V gradients use float32 atomic accumulation and are not bitwise deterministic.",
            "CUDA supports float16, bfloat16, and float32, window_radius <= 64, and attention dropout 0 only.",
            "The performance gate covers the long-context query path on the recorded GPU; small-shape launch overhead remains workload dependent.",
        ],
    }
    validate_cswa_evidence(report)
    return report


def validate_cswa_evidence(document: Mapping[str, Any]) -> None:
    """Validate schema, source identity, ordered runs, and the computed gate."""
    validate_common_evidence(
        document,
        component=COMPONENT,
        source_files=SOURCE_FILES,
        protocol=PROTOCOL,
        root=repository_root(),
    )
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    expected_order = [
        (_run_case_name(dtype_name, measurement), candidate)
        for dtype_name in DTYPES
        for measurement in MEASUREMENTS
        for candidate in ("reference", "cuda")
    ]
    actual_order = [(run["case"], run["candidate"]) for run in runs]
    if actual_order != expected_order:
        raise BenchmarkContractError(f"unexpected CSWA candidate order: {actual_order}")
    expected_decision = decide_cswa_profile(runs)
    if document["decision"] != expected_decision:
        raise BenchmarkContractError(
            "CSWA decision does not match measured gate inputs"
        )


def _semantic_signature(document: Mapping[str, Any]) -> dict[str, Any]:
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    decision = cast(Mapping[str, Any], document["decision"])
    cases = cast(Mapping[str, Mapping[str, Any]], decision["cases"])
    return {
        "component": document["component"],
        "protocol": document["protocol"],
        "source": {
            "files": cast(Mapping[str, Any], document["source"])["files"],
            "fingerprint_sha256": cast(Mapping[str, Any], document["source"])[
                "fingerprint_sha256"
            ],
        },
        "available": {
            f"{run['case']}::{run['candidate']}": run["available"] for run in runs
        },
        "parity": {
            f"{run['case']}::{run['candidate']}": cast(
                Mapping[str, Any], run["parity"]
            )["status"]
            for run in runs
        },
        "decision": decision["status"],
        "case_passed": {
            name: case_decision["passed"] for name, case_decision in cases.items()
        },
    }


def execute(
    *,
    evidence_path: Path,
    runtime_result_path: Path,
    record_evidence: bool,
) -> dict[str, Any]:
    """Write a fresh runtime result and record or validate stable evidence."""
    if evidence_path.resolve() == runtime_result_path.resolve():
        raise BenchmarkContractError("evidence and runtime-result paths must differ")
    runtime = profile_cswa()
    write_json_atomic(runtime_result_path, runtime)
    if record_evidence:
        write_json_atomic(evidence_path, runtime)
        return runtime
    stable = load_json_object(evidence_path)
    validate_cswa_evidence(stable)
    if _semantic_signature(stable) != _semantic_signature(runtime):
        raise BenchmarkContractError(
            "fresh runtime decision/parity/availability differs from stable evidence"
        )
    return runtime
