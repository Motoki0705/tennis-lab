"""Profile fixed-width mHC eager, compiled, and custom CUDA candidates.

The custom candidate is a benchmark-only handwritten Triton prototype for the
residual/write-back hot path.  It is deliberately not a production dispatch:
only a profile result that clears the speed, memory, and parity gates can
justify adding the separately owned production op package.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

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
from src.utils.models.components.mhc import (
    ManifoldConstrainedHyperConnection,
    MHCConfig,
)

COMPONENT = "mhc"
SOURCE_FILES = (
    "src/tasks/blcs/benchmarks/contracts.py",
    "src/tasks/blcs/benchmarks/track_query_mhc/__init__.py",
    "src/utils/models/components/mhc.py",
)
SEED = ISSUE_NUMBER
WARMUP = 3
ITERATIONS = 8
FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 1.0e-4
BACKWARD_ATOL = 2.0e-5
BACKWARD_RTOL = 2.0e-4
REQUIRED_SPEEDUP = 1.10
PROFILE_SHAPE: dict[str, int | float] = {
    "batch_size": 8,
    "num_views": 3,
    "frames": 512,
    "leading_rows": 8 * 3 * 512,
    "num_streams": 4,
    "dim": 64,
    "coefficient_dim": 64,
    "sinkhorn_iters": 20,
    "mask_density": 0.875,
}
PROTOCOL: dict[str, Any] = {
    "seed": SEED,
    "measurement": "forward+backward",
    "synchronize_each_iteration": True,
    "warmup": WARMUP,
    "iterations": ITERATIONS,
    "shape": PROFILE_SHAPE,
    "dtype": "float32",
    "update_scale": 0.5,
    "gate": {
        "minimum_speedup": REQUIRED_SPEEDUP,
        "memory_non_increase": True,
        "forward_atol": FORWARD_ATOL,
        "forward_rtol": FORWARD_RTOL,
        "backward_atol": BACKWARD_ATOL,
        "backward_rtol": BACKWARD_RTOL,
    },
}

_WORKER_PREFIX = "MHC_WORKER_RESULT="


class _MHCProfileModule(nn.Module):
    def __init__(
        self,
        mhc: ManifoldConstrainedHyperConnection,
        custom_post: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
        | None = None,
    ) -> None:
        super().__init__()
        self.mhc = mhc
        self.custom_post = custom_post

    def forward(self, streams: Tensor, valid_mask: Tensor) -> Tensor:
        projected, state = self.mhc.pre(streams, valid_mask)
        update = torch.tanh(projected)
        if self.custom_post is None:
            return self.mhc.post(update, streams, state)
        return self.custom_post(
            state.residual_mix,
            state.post_weights,
            update,
            streams,
            state.valid_mask,
            self.mhc.update_scale,
        )


def _mhc_config() -> MHCConfig:
    return MHCConfig(
        dim=int(PROFILE_SHAPE["dim"]),
        num_streams=int(PROFILE_SHAPE["num_streams"]),
        coefficient_dim=int(PROFILE_SHAPE["coefficient_dim"]),
        sinkhorn_iters=int(PROFILE_SHAPE["sinkhorn_iters"]),
        eps=1.0e-6,
        residual_identity_bias=4.0,
        update_scale_init=float(PROTOCOL["update_scale"]),
    )


def _make_inputs(leading_rows: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED)
    streams = torch.randn(
        leading_rows,
        int(PROFILE_SHAPE["num_streams"]),
        int(PROFILE_SHAPE["dim"]),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    mask = torch.rand(
        leading_rows,
        int(PROFILE_SHAPE["num_streams"]),
        device="cuda",
        generator=generator,
    ) < float(PROFILE_SHAPE["mask_density"])
    mask[0] = True
    if leading_rows > 1:
        mask[1] = False
    return streams, mask


def _benchmark_iteration(
    module: Callable[[Tensor, Tensor], Tensor],
    parameters: Sequence[nn.Parameter],
    streams: Tensor,
    valid_mask: Tensor,
) -> None:
    for parameter in parameters:
        parameter.grad = None
    streams.grad = None
    output = module(streams, valid_mask)
    loss = output.float().square().mean()
    loss.backward()


def _measure_candidate(
    module: Callable[[Tensor, Tensor], Tensor],
    parameters: Sequence[nn.Parameter],
    streams: Tensor,
    valid_mask: Tensor,
) -> tuple[dict[str, float], dict[str, int], dict[str, float | str]]:
    streams.requires_grad_(True)
    for _ in range(WARMUP):
        _benchmark_iteration(module, parameters, streams, valid_mask)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _benchmark_iteration(module, parameters, streams, valid_mask)
    torch.cuda.synchronize()
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    elapsed_ms: list[float] = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _benchmark_iteration(module, parameters, streams, valid_mask)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    elapsed_ms.sort()
    median = _percentile(elapsed_ms, 0.5)
    p95 = _percentile(elapsed_ms, 0.95)
    leading_rows = int(PROFILE_SHAPE["leading_rows"])
    return (
        {"median_ms": median, "p95_ms": p95},
        memory,
        {"unit": "stream-sets/s", "value": leading_rows / (median / 1000.0)},
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _tensor_error(reference: Tensor, actual: Tensor) -> tuple[float, float]:
    difference = (reference - actual).abs().float()
    return float(difference.max().item()), float(difference.mean().item())


def _parity_for_candidate(
    custom_post: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
    | None,
    *,
    compile_reference: bool,
) -> dict[str, Any]:
    torch.manual_seed(SEED)
    reference_model = _MHCProfileModule(
        ManifoldConstrainedHyperConnection(_mhc_config()).cuda()
    )
    candidate_model = copy.deepcopy(reference_model)
    candidate_model.custom_post = custom_post
    candidate_callable: Callable[[Tensor, Tensor], Tensor] = candidate_model
    if compile_reference:
        candidate_callable = torch.compile(
            candidate_model,
            fullgraph=False,
            dynamic=False,
            mode="reduce-overhead",
        )

    reference_streams, valid_mask = _make_inputs(17)
    candidate_streams = reference_streams.detach().clone().requires_grad_(True)
    reference_streams.requires_grad_(True)
    reference_output = reference_model(reference_streams, valid_mask)
    candidate_output = candidate_callable(candidate_streams, valid_mask)
    reference_output.square().sum().backward()
    candidate_output.square().sum().backward()

    forward_max, forward_mean = _tensor_error(reference_output, candidate_output)
    reference_stream_grad = reference_streams.grad
    candidate_stream_grad = candidate_streams.grad
    if reference_stream_grad is None or candidate_stream_grad is None:
        raise RuntimeError("mHC parity inputs did not receive gradients")
    backward_errors = [_tensor_error(reference_stream_grad, candidate_stream_grad)]
    parameters_close = True
    for reference_parameter, candidate_parameter in zip(
        reference_model.parameters(), candidate_model.parameters(), strict=True
    ):
        if reference_parameter.grad is None or candidate_parameter.grad is None:
            if reference_parameter.grad is candidate_parameter.grad:
                continue
            return {
                "status": "fail",
                "forward_max_abs_error": forward_max,
                "forward_mean_abs_error": forward_mean,
                "backward_max_abs_error": math.inf,
                "backward_mean_abs_error": math.inf,
                "atol": BACKWARD_ATOL,
                "rtol": BACKWARD_RTOL,
            }
        backward_errors.append(
            _tensor_error(reference_parameter.grad, candidate_parameter.grad)
        )
        parameters_close = parameters_close and torch.allclose(
            reference_parameter.grad,
            candidate_parameter.grad,
            atol=BACKWARD_ATOL,
            rtol=BACKWARD_RTOL,
        )
    backward_max = max(error[0] for error in backward_errors)
    backward_mean = max(error[1] for error in backward_errors)
    forward_close = torch.allclose(
        reference_output,
        candidate_output,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    backward_close = torch.allclose(
        reference_stream_grad,
        candidate_stream_grad,
        atol=BACKWARD_ATOL,
        rtol=BACKWARD_RTOL,
    )
    return {
        "status": (
            "pass" if forward_close and backward_close and parameters_close else "fail"
        ),
        "forward_max_abs_error": forward_max,
        "forward_mean_abs_error": forward_mean,
        "backward_max_abs_error": backward_max,
        "backward_mean_abs_error": backward_mean,
        "atol": BACKWARD_ATOL,
        "rtol": BACKWARD_RTOL,
    }


def _reference_self_parity() -> dict[str, Any]:
    return {
        "status": "pass",
        "forward_max_abs_error": 0.0,
        "forward_mean_abs_error": 0.0,
        "backward_max_abs_error": 0.0,
        "backward_mean_abs_error": 0.0,
        "atol": BACKWARD_ATOL,
        "rtol": BACKWARD_RTOL,
    }


def _available_run(
    *,
    candidate: str,
    kind: str,
    implementation: str,
    latency: Mapping[str, float],
    memory: Mapping[str, int],
    throughput: Mapping[str, float | str],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": "configured-training-lower-bound",
        "candidate": candidate,
        "kind": kind,
        "implementation": implementation,
        "available": True,
        "shape": PROFILE_SHAPE,
        "dtype": "float32",
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": dict(latency),
        "throughput": dict(throughput),
        "memory": dict(memory),
        "parity": dict(parity),
        "unavailable_reason": None,
    }


def _run_worker(candidate: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch CUDA is unavailable")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    custom_post = _build_triton_post() if candidate == "custom_cuda_prototype" else None
    profile_model = _MHCProfileModule(
        ManifoldConstrainedHyperConnection(_mhc_config()).cuda(),
        custom_post=custom_post,
    )
    profile_callable: Callable[[Tensor, Tensor], Tensor] = profile_model
    parity = _reference_self_parity()
    kind = "pytorch_reference"
    implementation = "eager_pytorch"
    if candidate == "compiled":
        profile_callable = torch.compile(
            profile_model,
            fullgraph=False,
            dynamic=False,
            mode="reduce-overhead",
        )
        parity = _parity_for_candidate(None, compile_reference=True)
        implementation = "torch_compile_reduce_overhead"
    elif candidate == "custom_cuda_prototype":
        parity = _parity_for_candidate(custom_post, compile_reference=False)
        kind = "cuda_prototype"
        implementation = "handwritten_triton_fused_post_forward_backward"
    elif candidate != "eager":
        raise ValueError(f"unsupported worker candidate: {candidate}")

    streams, valid_mask = _make_inputs(int(PROFILE_SHAPE["leading_rows"]))
    latency, memory, throughput = _measure_candidate(
        profile_callable,
        tuple(profile_model.parameters()),
        streams,
        valid_mask,
    )
    return _available_run(
        candidate=candidate,
        kind=kind,
        implementation=implementation,
        latency=latency,
        memory=memory,
        throughput=throughput,
        parity=parity,
    )


def _profile_subprocess(candidate: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.tasks.blcs.benchmarks.track_query_mhc",
        "--worker-candidate",
        candidate,
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
            case="configured-training-lower-bound",
            candidate=candidate,
            kind="cuda_prototype"
            if candidate == "custom_cuda_prototype"
            else "pytorch_reference",
            implementation=(
                "handwritten_triton_fused_post_forward_backward"
                if candidate == "custom_cuda_prototype"
                else "torch_compile_reduce_overhead"
            ),
            shape=PROFILE_SHAPE,
            dtype="float32",
            warmup=WARMUP,
            iterations=ITERATIONS,
            reason=reason[:1000],
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_PREFIX):
            payload = json.loads(line.removeprefix(_WORKER_PREFIX))
            if not isinstance(payload, dict):
                break
            return cast(dict[str, Any], payload)
    raise RuntimeError(f"worker {candidate} did not emit a result record")


def profile_mhc() -> dict[str, Any]:
    """Run every candidate in an isolated CUDA subprocess."""
    environment = build_cuda_environment()
    runs = [
        _profile_subprocess(candidate)
        for candidate in ("eager", "compiled", "custom_cuda_prototype")
    ]
    if not runs[0]["available"]:
        raise RuntimeError(
            f"eager PyTorch benchmark failed: {runs[0]['unavailable_reason']}"
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": COMPONENT,
        "generated_at_utc": utc_timestamp(),
        "source": build_source_record(repository_root(), SOURCE_FILES),
        "environment": environment,
        "protocol": PROTOCOL,
        "runs": runs,
        "decision": decide_mhc_profile(runs),
        "risks": [
            "The profile covers float32 and the canonical Q=4 width only.",
            "The prototype accelerates only residual/write-back; coefficient generation and masked Sinkhorn remain eager PyTorch.",
        ],
    }
    validate_mhc_evidence(report)
    return report


def decide_mhc_profile(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen 1.10x/parity/non-increasing-memory mHC GO gate."""
    references = [
        run
        for run in runs
        if run.get("kind") == "pytorch_reference"
        and run.get("available") is True
        and cast(Mapping[str, Any], run["parity"])["status"] == "pass"
    ]
    if not references:
        raise BenchmarkContractError(
            "at least one passing PyTorch reference is required"
        )
    best_reference = min(
        references,
        key=lambda run: float(cast(Mapping[str, Any], run["latency"])["median_ms"]),
    )
    optimized = next(
        (run for run in runs if run.get("candidate") == "custom_cuda_prototype"),
        None,
    )
    if optimized is None or optimized.get("available") is not True:
        reason = (
            "Custom CUDA prototype was unavailable; no production CUDA backend is registered."
            if optimized is None
            else f"Custom CUDA prototype unavailable: {optimized['unavailable_reason']}"
        )
        return {
            "status": "NO-GO",
            "best_reference": best_reference["candidate"],
            "optimized_candidate": None,
            "speedup": None,
            "memory_non_increase": False,
            "parity_passed": False,
            "required_speedup": REQUIRED_SPEEDUP,
            "reason": reason,
        }

    reference_latency = float(
        cast(Mapping[str, Any], best_reference["latency"])["median_ms"]
    )
    optimized_latency = float(
        cast(Mapping[str, Any], optimized["latency"])["median_ms"]
    )
    speedup = reference_latency / optimized_latency
    reference_memory_record = cast(Mapping[str, Any], best_reference["memory"])
    optimized_memory_record = cast(Mapping[str, Any], optimized["memory"])
    reference_allocated = int(reference_memory_record["peak_allocated_bytes"])
    optimized_allocated = int(optimized_memory_record["peak_allocated_bytes"])
    reference_reserved = int(reference_memory_record["peak_reserved_bytes"])
    optimized_reserved = int(optimized_memory_record["peak_reserved_bytes"])
    memory_non_increase = (
        optimized_allocated <= reference_allocated
        and optimized_reserved <= reference_reserved
    )
    parity_passed = cast(Mapping[str, Any], optimized["parity"])["status"] == "pass"
    passed = speedup >= REQUIRED_SPEEDUP and memory_non_increase and parity_passed
    failed_gates: list[str] = []
    if speedup < REQUIRED_SPEEDUP:
        failed_gates.append(f"speedup {speedup:.3f}x < {REQUIRED_SPEEDUP:.2f}x")
    if not memory_non_increase:
        failed_gates.append(
            "peak CUDA memory increased "
            f"(allocated {reference_allocated} -> {optimized_allocated} bytes, "
            f"reserved {reference_reserved} -> {optimized_reserved} bytes)"
        )
    if not parity_passed:
        failed_gates.append("forward/backward parity failed")
    return {
        "status": "GO" if passed else "NO-GO",
        "best_reference": best_reference["candidate"],
        "optimized_candidate": optimized["candidate"],
        "speedup": speedup,
        "memory_non_increase": memory_non_increase,
        "parity_passed": parity_passed,
        "required_speedup": REQUIRED_SPEEDUP,
        "reason": (
            "All fixed-width mHC CUDA gates passed."
            if passed
            else "No production CUDA backend is registered: "
            + "; ".join(failed_gates)
            + "."
        ),
    }


def validate_mhc_evidence(document: Mapping[str, Any]) -> None:
    """Validate the common schema and recompute the mHC decision exactly."""
    validate_common_evidence(
        document,
        component=COMPONENT,
        source_files=SOURCE_FILES,
        protocol=PROTOCOL,
        root=repository_root(),
    )
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    names = [run["candidate"] for run in runs]
    if names != ["eager", "compiled", "custom_cuda_prototype"]:
        raise BenchmarkContractError(f"unexpected mHC candidate order: {names}")
    expected_decision = decide_mhc_profile(runs)
    if document["decision"] != expected_decision:
        raise BenchmarkContractError("mHC decision does not match measured gate inputs")


def _semantic_signature(document: Mapping[str, Any]) -> dict[str, Any]:
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    return {
        "component": document["component"],
        "protocol": document["protocol"],
        "source": {
            "files": cast(Mapping[str, Any], document["source"])["files"],
            "fingerprint_sha256": cast(Mapping[str, Any], document["source"])[
                "fingerprint_sha256"
            ],
        },
        "available": {run["candidate"]: run["available"] for run in runs},
        "parity": {
            run["candidate"]: cast(Mapping[str, Any], run["parity"])["status"]
            for run in runs
        },
        "decision": cast(Mapping[str, Any], document["decision"])["status"],
    }


def execute(
    *, evidence_path: Path, runtime_result_path: Path, record_evidence: bool
) -> dict[str, Any]:
    """Write a fresh runtime result and record or validate stable evidence."""
    if evidence_path.resolve() == runtime_result_path.resolve():
        raise BenchmarkContractError("evidence and runtime-result paths must differ")
    runtime = profile_mhc()
    write_json_atomic(runtime_result_path, runtime)
    if record_evidence:
        write_json_atomic(evidence_path, runtime)
        return runtime

    stable = load_json_object(evidence_path)
    validate_mhc_evidence(stable)
    if _semantic_signature(stable) != _semantic_signature(runtime):
        raise BenchmarkContractError(
            "fresh runtime decision/parity/availability differs from stable evidence"
        )
    return runtime


def _build_triton_post() -> Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor
]:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    @triton.jit  # type: ignore[untyped-decorator]
    def forward_kernel(  # type: ignore[no-untyped-def]
        mix,
        post,
        update,
        residual,
        mask,
        scale,
        output,
        dim: tl.constexpr,
        streams: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        output_stream = tl.program_id(1)
        offsets = tl.arange(0, block_dim)
        feature_valid = offsets < dim
        output_valid = tl.load(mask + row * streams + output_stream)
        accumulated = tl.zeros((block_dim,), dtype=tl.float32)
        for input_stream in range(streams):
            input_valid = tl.load(mask + row * streams + input_stream)
            coefficient = tl.load(
                mix + (row * streams + output_stream) * streams + input_stream
            )
            values = tl.load(
                residual + (row * streams + input_stream) * dim + offsets,
                mask=feature_valid,
                other=0.0,
            )
            accumulated += coefficient * values * input_valid
        write_weight = tl.load(post + row * streams + output_stream)
        update_values = tl.load(
            update + row * dim + offsets,
            mask=feature_valid,
            other=0.0,
        )
        update_scale = tl.load(scale)
        result = (
            accumulated + update_scale * write_weight * update_values
        ) * output_valid
        tl.store(
            output + (row * streams + output_stream) * dim + offsets,
            result,
            mask=feature_valid,
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def grad_residual_kernel(  # type: ignore[no-untyped-def]
        grad_output,
        mix,
        mask,
        grad_residual,
        dim: tl.constexpr,
        streams: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        input_stream = tl.program_id(1)
        offsets = tl.arange(0, block_dim)
        feature_valid = offsets < dim
        input_valid = tl.load(mask + row * streams + input_stream)
        accumulated = tl.zeros((block_dim,), dtype=tl.float32)
        for output_stream in range(streams):
            output_valid = tl.load(mask + row * streams + output_stream)
            coefficient = tl.load(
                mix + (row * streams + output_stream) * streams + input_stream
            )
            gradients = tl.load(
                grad_output + (row * streams + output_stream) * dim + offsets,
                mask=feature_valid,
                other=0.0,
            )
            accumulated += gradients * coefficient * output_valid
        tl.store(
            grad_residual + (row * streams + input_stream) * dim + offsets,
            accumulated * input_valid,
            mask=feature_valid,
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def grad_mix_kernel(  # type: ignore[no-untyped-def]
        grad_output,
        residual,
        mask,
        grad_mix,
        dim: tl.constexpr,
        streams: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        output_stream = tl.program_id(1)
        input_stream = tl.program_id(2)
        offsets = tl.arange(0, block_dim)
        feature_valid = offsets < dim
        output_valid = tl.load(mask + row * streams + output_stream)
        input_valid = tl.load(mask + row * streams + input_stream)
        gradients = tl.load(
            grad_output + (row * streams + output_stream) * dim + offsets,
            mask=feature_valid,
            other=0.0,
        )
        values = tl.load(
            residual + (row * streams + input_stream) * dim + offsets,
            mask=feature_valid,
            other=0.0,
        )
        value = tl.sum(gradients * values, axis=0) * output_valid * input_valid
        tl.store(
            grad_mix + (row * streams + output_stream) * streams + input_stream,
            value,
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def grad_post_kernel(  # type: ignore[no-untyped-def]
        grad_output,
        update,
        mask,
        scale,
        grad_post,
        dim: tl.constexpr,
        streams: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        output_stream = tl.program_id(1)
        offsets = tl.arange(0, block_dim)
        feature_valid = offsets < dim
        output_valid = tl.load(mask + row * streams + output_stream)
        gradients = tl.load(
            grad_output + (row * streams + output_stream) * dim + offsets,
            mask=feature_valid,
            other=0.0,
        )
        values = tl.load(update + row * dim + offsets, mask=feature_valid, other=0.0)
        value = tl.sum(gradients * values, axis=0) * tl.load(scale) * output_valid
        tl.store(grad_post + row * streams + output_stream, value)

    @triton.jit  # type: ignore[untyped-decorator]
    def grad_update_kernel(  # type: ignore[no-untyped-def]
        grad_output,
        post,
        mask,
        scale,
        grad_update,
        dim: tl.constexpr,
        streams: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offsets = tl.arange(0, block_dim)
        feature_valid = offsets < dim
        accumulated = tl.zeros((block_dim,), dtype=tl.float32)
        for output_stream in range(streams):
            output_valid = tl.load(mask + row * streams + output_stream)
            write_weight = tl.load(post + row * streams + output_stream)
            gradients = tl.load(
                grad_output + (row * streams + output_stream) * dim + offsets,
                mask=feature_valid,
                other=0.0,
            )
            accumulated += gradients * write_weight * output_valid
        tl.store(
            grad_update + row * dim + offsets,
            accumulated * tl.load(scale),
            mask=feature_valid,
        )

    class _TritonMHCPost(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            mix: Tensor,
            post: Tensor,
            update: Tensor,
            residual: Tensor,
            mask: Tensor,
            scale: Tensor,
        ) -> Tensor:
            if mix.dtype != torch.float32 or residual.dtype != torch.float32:
                raise TypeError("mHC Triton prototype supports float32 only")
            if not all(
                tensor.is_cuda for tensor in (mix, post, update, residual, mask, scale)
            ):
                raise ValueError("mHC Triton prototype requires CUDA tensors")
            num_streams, dim = residual.shape[-2:]
            leading_rows = residual.numel() // (num_streams * dim)
            flat_mix = mix.reshape(leading_rows, num_streams, num_streams).contiguous()
            flat_post = post.reshape(leading_rows, num_streams).contiguous()
            flat_update = update.reshape(leading_rows, dim).contiguous()
            flat_residual = residual.reshape(
                leading_rows, num_streams, dim
            ).contiguous()
            flat_mask = mask.reshape(leading_rows, num_streams).contiguous()
            output = torch.empty_like(flat_residual)
            block_dim = triton.next_power_of_2(dim)
            forward_kernel[(leading_rows, num_streams)](
                flat_mix,
                flat_post,
                flat_update,
                flat_residual,
                flat_mask,
                scale,
                output,
                dim=dim,
                streams=num_streams,
                block_dim=block_dim,
            )
            ctx.save_for_backward(
                flat_mix, flat_post, flat_update, flat_residual, flat_mask, scale
            )
            ctx.leading_shape = residual.shape[:-2]
            ctx.num_streams = num_streams
            ctx.dim = dim
            return output.reshape_as(residual)

        @staticmethod
        def backward(
            ctx: Any, grad_output: Tensor
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, None, Tensor]:
            flat_mix, flat_post, flat_update, flat_residual, flat_mask, scale = (
                ctx.saved_tensors
            )
            leading_rows = flat_residual.shape[0]
            num_streams = ctx.num_streams
            dim = ctx.dim
            flat_grad_output = grad_output.reshape(
                leading_rows, num_streams, dim
            ).contiguous()
            grad_mix = torch.empty_like(flat_mix)
            grad_post = torch.empty_like(flat_post)
            grad_update = torch.empty_like(flat_update)
            grad_residual = torch.empty_like(flat_residual)
            block_dim = triton.next_power_of_2(dim)
            grad_residual_kernel[(leading_rows, num_streams)](
                flat_grad_output,
                flat_mix,
                flat_mask,
                grad_residual,
                dim=dim,
                streams=num_streams,
                block_dim=block_dim,
            )
            grad_mix_kernel[(leading_rows, num_streams, num_streams)](
                flat_grad_output,
                flat_residual,
                flat_mask,
                grad_mix,
                dim=dim,
                streams=num_streams,
                block_dim=block_dim,
            )
            grad_post_kernel[(leading_rows, num_streams)](
                flat_grad_output,
                flat_update,
                flat_mask,
                scale,
                grad_post,
                dim=dim,
                streams=num_streams,
                block_dim=block_dim,
            )
            grad_update_kernel[(leading_rows,)](
                flat_grad_output,
                flat_post,
                flat_mask,
                scale,
                grad_update,
                dim=dim,
                streams=num_streams,
                block_dim=block_dim,
            )
            grad_scale = (
                (
                    flat_grad_output
                    * flat_mask.unsqueeze(-1)
                    * flat_post.unsqueeze(-1)
                    * flat_update.unsqueeze(1)
                )
                .sum()
                .reshape_as(scale)
            )
            leading_shape = ctx.leading_shape
            return (
                grad_mix.reshape(*leading_shape, num_streams, num_streams),
                grad_post.reshape(*leading_shape, num_streams, 1),
                grad_update.reshape(*leading_shape, 1, dim),
                grad_residual.reshape(*leading_shape, num_streams, dim),
                None,
                grad_scale,
            )

    def apply_post(
        mix: Tensor,
        post: Tensor,
        update: Tensor,
        residual: Tensor,
        mask: Tensor,
        scale: Tensor,
    ) -> Tensor:
        return cast(
            Tensor,
            _TritonMHCPost.apply(mix, post, update, residual, mask, scale),
        )

    return apply_post
