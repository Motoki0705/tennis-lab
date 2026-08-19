"""Profile token-compressor eager, compiled, and fused CUDA candidates.

The custom candidate is a benchmark-only Triton prototype.  Linear projections
remain PyTorch GEMMs; the prototype fuses the post-projection previous/current
gather, channel-wise masked softmax, and weighted reduction.  Production CUDA
dispatch is justified only when every target shape clears the frozen gate.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
from src.utils.models.components.compressor import (
    TokenLevelCompressorConfig,
    TokenLevelKVCompressor,
)

COMPONENT = "compressor"
SOURCE_FILES = (
    "src/tasks/blcs/benchmarks/contracts.py",
    "src/tasks/blcs/benchmarks/track_query_compressor/__init__.py",
    "src/utils/models/components/compressor.py",
)
SEED = ISSUE_NUMBER
WARMUP = 3
ITERATIONS = 6
FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 1.0e-4
BACKWARD_ATOL = 2.0e-5
BACKWARD_RTOL = 2.0e-4
REQUIRED_SPEEDUP = 1.10
MASK_DENSITY = 0.875


@dataclass(frozen=True, slots=True)
class _ProfileCase:
    name: str
    batch_size: int
    sequence_length: int
    dim: int
    n_heads: int
    compression_ratio: int = 4

    @property
    def shape(self) -> dict[str, int | float]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "dim": self.dim,
            "n_heads": self.n_heads,
            "head_dim": self.dim // self.n_heads,
            "compression_ratio": self.compression_ratio,
            "mask_density": MASK_DENSITY,
        }


PROFILE_CASES = (
    _ProfileCase(
        name="configured-default-object-path",
        batch_size=24,
        sequence_length=512,
        dim=64,
        n_heads=4,
    ),
    _ProfileCase(
        name="configured-small-query-long-context",
        batch_size=32,
        sequence_length=1025,
        dim=256,
        n_heads=4,
    ),
)
PARITY_CASES = (
    "first-block-short-sequence",
    "partial-tail",
    "all-invalid",
    "sparse-mask",
)
PROTOCOL: dict[str, Any] = {
    "seed": SEED,
    "measurement": "forward+backward",
    "synchronize_each_iteration": True,
    "warmup": WARMUP,
    "iterations": ITERATIONS,
    "target_shapes": [case.shape for case in PROFILE_CASES],
    "dtype": "float32",
    "mask_density": MASK_DENSITY,
    "parity_cases": list(PARITY_CASES),
    "prototype_scope": (
        "post-projection previous/current gather plus channel-wise masked "
        "softmax/reduction"
    ),
    "gate": {
        "minimum_speedup_each_shape": REQUIRED_SPEEDUP,
        "memory_non_increase_each_shape": True,
        "forward_atol": FORWARD_ATOL,
        "forward_rtol": FORWARD_RTOL,
        "backward_atol": BACKWARD_ATOL,
        "backward_rtol": BACKWARD_RTOL,
    },
}

_WORKER_PREFIX = "COMPRESSOR_WORKER_RESULT="
_ProfileOutput = tuple[Tensor, Tensor, Tensor, Tensor]
_Pool = Callable[[Tensor, Tensor, Tensor, int], tuple[Tensor, Tensor]]


def _compressor_config(case: _ProfileCase) -> TokenLevelCompressorConfig:
    return TokenLevelCompressorConfig(
        dim=case.dim,
        n_heads=case.n_heads,
        head_dim=case.dim // case.n_heads,
        compression_ratio=case.compression_ratio,
        overlap=True,
    )


class _CompressorProfileModule(nn.Module):
    """Expose a tuple output and optionally replace only post-projection pooling."""

    def __init__(
        self,
        compressor: TokenLevelKVCompressor,
        custom_pool: _Pool | None = None,
    ) -> None:
        super().__init__()
        self.compressor = compressor
        self.custom_pool = custom_pool
        self.register_forward_pre_hook(self._validate_forward_inputs)

    def _validate_forward_inputs(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
    ) -> None:
        self.compressor.validate_inputs(
            cast(Tensor, args[0]),
            cast(Tensor, args[1]),
        )

    def forward(self, x: Tensor, state_valid: Tensor) -> _ProfileOutput:
        if self.custom_pool is None:
            output = self.compressor(x, state_valid)
            return output.key, output.value, output.state_valid, output.positions

        batch_size, sequence_length, _ = x.shape
        masked_x = torch.where(state_valid.unsqueeze(-1), x, torch.zeros_like(x))
        raw_kv = self.compressor._project(masked_x, self.compressor.w_kv).reshape(
            batch_size,
            sequence_length,
            self.compressor.branches,
            self.compressor.kv_dim,
        )
        raw_gate = self.compressor._project(masked_x, self.compressor.w_gate).reshape(
            batch_size,
            sequence_length,
            self.compressor.branches,
            self.compressor.kv_dim,
        )
        offsets = (
            torch.arange(sequence_length, device=x.device)
            % self.compressor.compression_ratio
        )
        positional_gate = self.compressor.ape.index_select(0, offsets).reshape(
            sequence_length,
            self.compressor.branches,
            self.compressor.kv_dim,
        )
        raw_gate = raw_gate.float() + positional_gate.float().unsqueeze(0)
        compressed, compressed_valid = self.custom_pool(
            raw_kv.float(),
            raw_gate,
            state_valid,
            self.compressor.compression_ratio,
        )
        compressed = compressed.to(dtype=x.dtype)
        compressed_length = compressed.shape[1]
        split = compressed.reshape(
            batch_size,
            compressed_length,
            2,
            self.compressor.n_heads,
            self.compressor.head_dim,
        )
        key, value = split.unbind(dim=2)
        positions = (
            torch.arange(
                compressed_length,
                device=x.device,
                dtype=torch.float32,
            )
            * self.compressor.compression_ratio
            + (self.compressor.compression_ratio - 1) / 2
        ).clamp_max(float(sequence_length - 1))
        return (
            key.transpose(1, 2),
            value.transpose(1, 2),
            compressed_valid,
            positions,
        )


def _make_inputs(case: _ProfileCase) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + case.sequence_length + case.dim)
    x = torch.randn(
        case.batch_size,
        case.sequence_length,
        case.dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    state_valid = (
        torch.rand(
            case.batch_size,
            case.sequence_length,
            device="cuda",
            generator=generator,
        )
        < MASK_DENSITY
    )
    state_valid[0] = True
    if case.batch_size > 1:
        state_valid[1] = False
    return x, state_valid


def _benchmark_iteration(
    module: Callable[[Tensor, Tensor], _ProfileOutput],
    parameters: Sequence[nn.Parameter],
    x: Tensor,
    state_valid: Tensor,
) -> None:
    for parameter in parameters:
        parameter.grad = None
    x.grad = None
    key, value, _, _ = module(x, state_valid)
    loss = key.float().square().mean() + value.float().square().mean()
    loss.backward()


def _measure_candidate(
    module: Callable[[Tensor, Tensor], _ProfileOutput],
    parameters: Sequence[nn.Parameter],
    x: Tensor,
    state_valid: Tensor,
) -> tuple[dict[str, float], dict[str, int], dict[str, float | str]]:
    x.requires_grad_(True)
    for _ in range(WARMUP):
        _benchmark_iteration(module, parameters, x, state_valid)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _benchmark_iteration(module, parameters, x, state_valid)
    torch.cuda.synchronize()
    memory = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    elapsed_ms: list[float] = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _benchmark_iteration(module, parameters, x, state_valid)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    elapsed_ms.sort()
    median = _percentile(elapsed_ms, 0.5)
    p95 = _percentile(elapsed_ms, 0.95)
    tokens = x.shape[0] * x.shape[1]
    return (
        {"median_ms": median, "p95_ms": p95},
        memory,
        {"unit": "input-tokens/s", "value": tokens / (median / 1000.0)},
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _tensor_error(reference: Tensor, actual: Tensor) -> tuple[float, float]:
    difference = (reference - actual).abs().float()
    if difference.numel() == 0:
        return 0.0, 0.0
    return float(difference.max().item()), float(difference.mean().item())


def _parity_inputs(name: str) -> tuple[_ProfileCase, Tensor, Tensor]:
    if name == "first-block-short-sequence":
        case = _ProfileCase(
            name=name, batch_size=2, sequence_length=2, dim=16, n_heads=4
        )
        mask = torch.tensor([[True, True], [False, True]], device="cuda")
    elif name == "partial-tail":
        case = _ProfileCase(
            name=name, batch_size=2, sequence_length=7, dim=16, n_heads=4
        )
        mask = torch.ones(2, 7, dtype=torch.bool, device="cuda")
    elif name == "all-invalid":
        case = _ProfileCase(
            name=name, batch_size=2, sequence_length=8, dim=16, n_heads=4
        )
        mask = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
    elif name == "sparse-mask":
        case = _ProfileCase(
            name=name, batch_size=2, sequence_length=9, dim=16, n_heads=4
        )
        mask = torch.tensor(
            [
                [True, False, False, True, False, False, False, True, False],
                [False, False, True, False, False, True, False, False, True],
            ],
            device="cuda",
        )
    else:
        raise ValueError(f"unknown compressor parity case: {name}")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(SEED + case.sequence_length)
    x = torch.randn(
        case.batch_size,
        case.sequence_length,
        case.dim,
        device="cuda",
        generator=generator,
    )
    return case, x, mask


def _parity_for_candidate(
    custom_pool: _Pool | None,
    *,
    compile_reference: bool,
) -> dict[str, Any]:
    forward_errors: list[tuple[float, float]] = []
    backward_errors: list[tuple[float, float]] = []
    all_close = True
    for parity_case in PARITY_CASES:
        case, reference_x, state_valid = _parity_inputs(parity_case)
        torch.manual_seed(SEED + case.sequence_length)
        reference_model = _CompressorProfileModule(
            TokenLevelKVCompressor(_compressor_config(case)).cuda()
        )
        with torch.no_grad():
            reference_model.compressor.w_gate.weight.normal_(std=0.05)
            reference_model.compressor.w_gate.bias.normal_(std=0.03)
            reference_model.compressor.ape.normal_(std=0.08)
        candidate_model = copy.deepcopy(reference_model)
        candidate_model.custom_pool = custom_pool
        candidate_callable: Callable[[Tensor, Tensor], _ProfileOutput] = candidate_model
        if compile_reference:
            candidate_callable = torch.compile(
                candidate_model,
                fullgraph=False,
                dynamic=False,
                mode="reduce-overhead",
            )

        candidate_x = reference_x.detach().clone().requires_grad_(True)
        reference_x.requires_grad_(True)
        reference_output = reference_model(reference_x, state_valid)
        candidate_output = candidate_callable(candidate_x, state_valid)
        reference_loss = (
            reference_output[0].float().square().sum()
            + 0.7 * reference_output[1].float().square().sum()
        )
        candidate_loss = (
            candidate_output[0].float().square().sum()
            + 0.7 * candidate_output[1].float().square().sum()
        )
        reference_loss.backward()
        candidate_loss.backward()

        for expected, actual in zip(
            reference_output[:2], candidate_output[:2], strict=True
        ):
            forward_errors.append(_tensor_error(expected, actual))
            all_close = all_close and torch.allclose(
                expected,
                actual,
                atol=FORWARD_ATOL,
                rtol=FORWARD_RTOL,
            )
        all_close = all_close and torch.equal(reference_output[2], candidate_output[2])
        all_close = all_close and torch.equal(reference_output[3], candidate_output[3])

        if reference_x.grad is None or candidate_x.grad is None:
            raise RuntimeError("compressor parity inputs did not receive gradients")
        backward_errors.append(_tensor_error(reference_x.grad, candidate_x.grad))
        all_close = all_close and torch.allclose(
            reference_x.grad,
            candidate_x.grad,
            atol=BACKWARD_ATOL,
            rtol=BACKWARD_RTOL,
        )
        for reference_parameter, candidate_parameter in zip(
            reference_model.parameters(), candidate_model.parameters(), strict=True
        ):
            if reference_parameter.grad is None or candidate_parameter.grad is None:
                all_close = all_close and (
                    reference_parameter.grad is candidate_parameter.grad
                )
                continue
            backward_errors.append(
                _tensor_error(reference_parameter.grad, candidate_parameter.grad)
            )
            all_close = all_close and torch.allclose(
                reference_parameter.grad,
                candidate_parameter.grad,
                atol=BACKWARD_ATOL,
                rtol=BACKWARD_RTOL,
            )

    forward_max = max(error[0] for error in forward_errors)
    forward_mean = max(error[1] for error in forward_errors)
    backward_max = max(error[0] for error in backward_errors)
    backward_mean = max(error[1] for error in backward_errors)
    return {
        "status": "pass" if all_close else "fail",
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
    case: _ProfileCase,
    candidate: str,
    kind: str,
    implementation: str,
    latency: Mapping[str, float],
    memory: Mapping[str, int],
    throughput: Mapping[str, float | str],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": case.name,
        "candidate": candidate,
        "kind": kind,
        "implementation": implementation,
        "available": True,
        "shape": case.shape,
        "dtype": "float32",
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": dict(latency),
        "throughput": dict(throughput),
        "memory": dict(memory),
        "parity": dict(parity),
        "unavailable_reason": None,
    }


def _run_worker(candidate: str, case_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch CUDA is unavailable")
    case = PROFILE_CASES[case_index]
    torch.manual_seed(SEED + case_index)
    torch.cuda.manual_seed_all(SEED + case_index)
    custom_pool = _build_triton_pool() if candidate == "custom_cuda_prototype" else None
    profile_model = _CompressorProfileModule(
        TokenLevelKVCompressor(_compressor_config(case)).cuda(),
        custom_pool=custom_pool,
    )
    profile_callable: Callable[[Tensor, Tensor], _ProfileOutput] = profile_model
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
        parity = _parity_for_candidate(custom_pool, compile_reference=False)
        kind = "cuda_prototype"
        implementation = "handwritten_triton_fused_gather_softmax_reduce"
    elif candidate != "eager":
        raise ValueError(f"unsupported worker candidate: {candidate}")

    x, state_valid = _make_inputs(case)
    latency, memory, throughput = _measure_candidate(
        profile_callable,
        tuple(profile_model.parameters()),
        x,
        state_valid,
    )
    return _available_run(
        case=case,
        candidate=candidate,
        kind=kind,
        implementation=implementation,
        latency=latency,
        memory=memory,
        throughput=throughput,
        parity=parity,
    )


def _profile_subprocess(candidate: str, case_index: int) -> dict[str, Any]:
    case = PROFILE_CASES[case_index]
    command = [
        sys.executable,
        "-m",
        "src.tasks.blcs.benchmarks.track_query_compressor",
        "--worker-candidate",
        candidate,
        "--worker-case",
        str(case_index),
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
            case=case.name,
            candidate=candidate,
            kind=(
                "cuda_prototype"
                if candidate == "custom_cuda_prototype"
                else "pytorch_reference"
            ),
            implementation=(
                "handwritten_triton_fused_gather_softmax_reduce"
                if candidate == "custom_cuda_prototype"
                else (
                    "torch_compile_reduce_overhead"
                    if candidate == "compiled"
                    else "eager_pytorch"
                )
            ),
            shape=case.shape,
            dtype="float32",
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
    raise RuntimeError(f"worker {candidate}/{case.name} did not emit a result record")


def profile_compressor() -> dict[str, Any]:
    """Run every candidate and target shape in isolated CUDA subprocesses."""
    environment = build_cuda_environment()
    runs = [
        _profile_subprocess(candidate, case_index)
        for case_index in range(len(PROFILE_CASES))
        for candidate in ("eager", "compiled", "custom_cuda_prototype")
    ]
    eager_runs = [run for run in runs if run["candidate"] == "eager"]
    if any(not run["available"] for run in eager_runs):
        reasons = [
            run["unavailable_reason"] for run in eager_runs if not run["available"]
        ]
        raise RuntimeError(f"eager PyTorch benchmark failed: {reasons}")
    report: dict[str, Any] = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": COMPONENT,
        "generated_at_utc": utc_timestamp(),
        "source": build_source_record(repository_root(), SOURCE_FILES),
        "environment": environment,
        "protocol": PROTOCOL,
        "runs": runs,
        "decision": decide_compressor_profile(runs),
        "risks": [
            "The profile covers float32 with compression ratio 4; mixed precision remains unevaluated.",
            "The benchmark-only prototype supports contiguous post-projection tensors and does not provide production dispatch.",
        ],
    }
    validate_compressor_evidence(report)
    return report


def decide_compressor_profile(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the per-shape 1.10x/parity/non-increasing-memory gate."""
    case_decisions: dict[str, dict[str, Any]] = {}
    failed_reasons: list[str] = []
    for case in PROFILE_CASES:
        case_runs = [run for run in runs if run.get("case") == case.name]
        references = [
            run
            for run in case_runs
            if run.get("kind") == "pytorch_reference"
            and run.get("available") is True
            and cast(Mapping[str, Any], run["parity"])["status"] == "pass"
        ]
        if not references:
            raise BenchmarkContractError(
                f"at least one passing PyTorch reference is required for {case.name}"
            )
        best_reference = min(
            references,
            key=lambda run: float(cast(Mapping[str, Any], run["latency"])["median_ms"]),
        )
        optimized = next(
            (
                run
                for run in case_runs
                if run.get("candidate") == "custom_cuda_prototype"
            ),
            None,
        )
        if optimized is None or optimized.get("available") is not True:
            reason = (
                "custom CUDA prototype missing"
                if optimized is None
                else f"custom CUDA prototype unavailable: {optimized['unavailable_reason']}"
            )
            case_decisions[case.name] = {
                "best_reference": best_reference["candidate"],
                "speedup": None,
                "memory_non_increase": False,
                "parity_passed": False,
                "passed": False,
            }
            failed_reasons.append(f"{case.name}: {reason}")
            continue

        reference_latency = float(
            cast(Mapping[str, Any], best_reference["latency"])["median_ms"]
        )
        optimized_latency = float(
            cast(Mapping[str, Any], optimized["latency"])["median_ms"]
        )
        speedup = reference_latency / optimized_latency
        reference_memory = cast(Mapping[str, Any], best_reference["memory"])
        optimized_memory = cast(Mapping[str, Any], optimized["memory"])
        memory_non_increase = all(
            int(optimized_memory[key]) <= int(reference_memory[key])
            for key in ("peak_allocated_bytes", "peak_reserved_bytes")
        )
        parity_passed = cast(Mapping[str, Any], optimized["parity"])["status"] == "pass"
        passed = speedup >= REQUIRED_SPEEDUP and memory_non_increase and parity_passed
        case_decisions[case.name] = {
            "best_reference": best_reference["candidate"],
            "speedup": speedup,
            "memory_non_increase": memory_non_increase,
            "parity_passed": parity_passed,
            "passed": passed,
        }
        gates: list[str] = []
        if speedup < REQUIRED_SPEEDUP:
            gates.append(f"speedup {speedup:.3f}x < {REQUIRED_SPEEDUP:.2f}x")
        if not memory_non_increase:
            gates.append("peak CUDA memory increased")
        if not parity_passed:
            gates.append("mask-matrix forward/backward parity failed")
        if gates:
            failed_reasons.append(f"{case.name}: {', '.join(gates)}")

    passed_all = all(case["passed"] for case in case_decisions.values())
    return {
        "status": "GO" if passed_all else "NO-GO",
        "required_speedup": REQUIRED_SPEEDUP,
        "optimized_candidate": ("custom_cuda_prototype" if passed_all else None),
        "cases": case_decisions,
        "reason": (
            "All token-compressor CUDA gates passed for every target shape."
            if passed_all
            else "No production CUDA backend is registered: "
            + "; ".join(failed_reasons)
            + "."
        ),
    }


def validate_compressor_evidence(document: Mapping[str, Any]) -> None:
    """Validate the common schema and recompute the compressor decision."""
    validate_common_evidence(
        document,
        component=COMPONENT,
        source_files=SOURCE_FILES,
        protocol=PROTOCOL,
        root=repository_root(),
    )
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    expected_order = [
        (case.name, candidate)
        for case in PROFILE_CASES
        for candidate in ("eager", "compiled", "custom_cuda_prototype")
    ]
    actual_order = [(run["case"], run["candidate"]) for run in runs]
    if actual_order != expected_order:
        raise BenchmarkContractError(
            f"unexpected compressor candidate order: {actual_order}"
        )
    expected_decision = decide_compressor_profile(runs)
    if document["decision"] != expected_decision:
        raise BenchmarkContractError(
            "compressor decision does not match measured gate inputs"
        )


def _semantic_signature(document: Mapping[str, Any]) -> dict[str, Any]:
    runs = cast(Sequence[Mapping[str, Any]], document["runs"])
    decision = cast(Mapping[str, Any], document["decision"])
    case_decisions = cast(Mapping[str, Mapping[str, Any]], decision["cases"])
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
            name: case_decision["passed"]
            for name, case_decision in case_decisions.items()
        },
    }


def execute(
    *, evidence_path: Path, runtime_result_path: Path, record_evidence: bool
) -> dict[str, Any]:
    """Write a fresh runtime result and record or validate stable evidence."""
    if evidence_path.resolve() == runtime_result_path.resolve():
        raise BenchmarkContractError("evidence and runtime-result paths must differ")
    runtime = profile_compressor()
    write_json_atomic(runtime_result_path, runtime)
    if record_evidence:
        write_json_atomic(evidence_path, runtime)
        return runtime

    stable = load_json_object(evidence_path)
    validate_compressor_evidence(stable)
    if _semantic_signature(stable) != _semantic_signature(runtime):
        raise BenchmarkContractError(
            "fresh runtime decision/parity/availability differs from stable evidence"
        )
    return runtime


def _build_triton_pool() -> _Pool:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    @triton.jit  # type: ignore[untyped-decorator]
    def forward_kernel(  # type: ignore[no-untyped-def]
        raw_kv,
        raw_gate,
        state_valid,
        output,
        output_valid,
        sequence_length: tl.constexpr,
        compressed_length: tl.constexpr,
        kv_dim: tl.constexpr,
        ratio: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        batch = row // compressed_length
        compressed_index = row - batch * compressed_length
        channels = tl.arange(0, block_dim)
        channel_valid = channels < kv_dim
        maximum = tl.full((block_dim,), -float("inf"), tl.float32)
        any_valid = False
        for source in range(2 * ratio):
            token = (compressed_index - 1) * ratio + source
            branch: tl.constexpr = source // ratio
            boundary_valid = (token >= 0) & (token < sequence_length)
            safe_token = tl.maximum(0, tl.minimum(token, sequence_length - 1))
            valid = boundary_valid & tl.load(
                state_valid + batch * sequence_length + safe_token
            )
            gate = tl.load(
                raw_gate
                + ((batch * sequence_length + safe_token) * 2 + branch) * kv_dim
                + channels,
                mask=channel_valid,
                other=0.0,
            )
            maximum = tl.maximum(maximum, tl.where(valid, gate, -float("inf")))
            any_valid = any_valid | valid
        maximum = tl.where(any_valid, maximum, 0.0)
        denominator = tl.zeros((block_dim,), tl.float32)
        numerator = tl.zeros((block_dim,), tl.float32)
        for source in range(2 * ratio):
            token = (compressed_index - 1) * ratio + source
            reduction_branch: tl.constexpr = source // ratio
            boundary_valid = (token >= 0) & (token < sequence_length)
            safe_token = tl.maximum(0, tl.minimum(token, sequence_length - 1))
            valid = boundary_valid & tl.load(
                state_valid + batch * sequence_length + safe_token
            )
            gate = tl.load(
                raw_gate
                + ((batch * sequence_length + safe_token) * 2 + reduction_branch)
                * kv_dim
                + channels,
                mask=channel_valid,
                other=0.0,
            )
            values = tl.load(
                raw_kv
                + ((batch * sequence_length + safe_token) * 2 + reduction_branch)
                * kv_dim
                + channels,
                mask=channel_valid,
                other=0.0,
            )
            weight = tl.where(valid, tl.exp(gate - maximum), 0.0)
            denominator += weight
            numerator += weight * values
        result = tl.where(any_valid, numerator / denominator, 0.0)
        tl.store(
            output + row * kv_dim + channels,
            result,
            mask=channel_valid,
        )
        tl.store(output_valid + row, any_valid)

    @triton.jit  # type: ignore[untyped-decorator]
    def backward_kernel(  # type: ignore[no-untyped-def]
        grad_output,
        raw_kv,
        raw_gate,
        state_valid,
        output,
        grad_raw_kv,
        grad_raw_gate,
        sequence_length: tl.constexpr,
        compressed_length: tl.constexpr,
        kv_dim: tl.constexpr,
        ratio: tl.constexpr,
        block_dim: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        batch = row // (sequence_length * 2)
        within_batch = row - batch * sequence_length * 2
        token = within_batch // 2
        branch = within_batch - token * 2
        block_index = token // ratio
        compressed_index = block_index + (1 - branch)
        channels = tl.arange(0, block_dim)
        channel_valid = channels < kv_dim
        contributes = (compressed_index < compressed_length) & tl.load(
            state_valid + batch * sequence_length + token
        )
        maximum = tl.full((block_dim,), -float("inf"), tl.float32)
        for source in range(2 * ratio):
            source_token = (compressed_index - 1) * ratio + source
            source_branch: tl.constexpr = source // ratio
            boundary_valid = (source_token >= 0) & (source_token < sequence_length)
            safe_token = tl.maximum(0, tl.minimum(source_token, sequence_length - 1))
            valid = boundary_valid & tl.load(
                state_valid + batch * sequence_length + safe_token
            )
            gate = tl.load(
                raw_gate
                + ((batch * sequence_length + safe_token) * 2 + source_branch) * kv_dim
                + channels,
                mask=channel_valid,
                other=0.0,
            )
            maximum = tl.maximum(maximum, tl.where(valid, gate, -float("inf")))
        maximum = tl.where(contributes, maximum, 0.0)
        denominator = tl.zeros((block_dim,), tl.float32)
        for source in range(2 * ratio):
            source_token = (compressed_index - 1) * ratio + source
            denominator_branch: tl.constexpr = source // ratio
            boundary_valid = (source_token >= 0) & (source_token < sequence_length)
            safe_token = tl.maximum(0, tl.minimum(source_token, sequence_length - 1))
            valid = boundary_valid & tl.load(
                state_valid + batch * sequence_length + safe_token
            )
            gate = tl.load(
                raw_gate
                + ((batch * sequence_length + safe_token) * 2 + denominator_branch)
                * kv_dim
                + channels,
                mask=channel_valid,
                other=0.0,
            )
            denominator += tl.where(valid, tl.exp(gate - maximum), 0.0)
        raw_offset = ((batch * sequence_length + token) * 2 + branch) * kv_dim
        gate = tl.load(
            raw_gate + raw_offset + channels,
            mask=channel_valid,
            other=0.0,
        )
        values = tl.load(
            raw_kv + raw_offset + channels,
            mask=channel_valid,
            other=0.0,
        )
        output_offset = (batch * compressed_length + compressed_index) * kv_dim
        safe_output_offset = tl.where(contributes, output_offset, 0)
        pooled = tl.load(
            output + safe_output_offset + channels,
            mask=channel_valid,
            other=0.0,
        )
        upstream = tl.load(
            grad_output + safe_output_offset + channels,
            mask=channel_valid,
            other=0.0,
        )
        weight = tl.where(
            contributes,
            tl.exp(gate - maximum) / denominator,
            0.0,
        )
        tl.store(
            grad_raw_kv + raw_offset + channels,
            upstream * weight,
            mask=channel_valid,
        )
        tl.store(
            grad_raw_gate + raw_offset + channels,
            upstream * weight * (values - pooled),
            mask=channel_valid,
        )

    class _TritonCompressorPool(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            raw_kv: Tensor,
            raw_gate: Tensor,
            state_valid: Tensor,
            ratio: int,
        ) -> tuple[Tensor, Tensor]:
            if raw_kv.dtype != torch.float32 or raw_gate.dtype != torch.float32:
                raise TypeError("compressor Triton prototype supports float32 only")
            if not all(tensor.is_cuda for tensor in (raw_kv, raw_gate, state_valid)):
                raise ValueError("compressor Triton prototype requires CUDA tensors")
            if raw_kv.shape != raw_gate.shape or raw_kv.ndim != 4:
                raise ValueError("raw_kv and raw_gate must share shape [N,T,2,KVDim]")
            if raw_kv.shape[2] != 2 or state_valid.shape != raw_kv.shape[:2]:
                raise ValueError("compressor Triton prototype received invalid shapes")
            if not raw_kv.is_contiguous() or not raw_gate.is_contiguous():
                raise ValueError(
                    "compressor Triton prototype requires contiguous projections"
                )
            batch_size, sequence_length, _, kv_dim = raw_kv.shape
            compressed_length = (sequence_length + ratio - 1) // ratio
            output = torch.empty(
                batch_size,
                compressed_length,
                kv_dim,
                device=raw_kv.device,
                dtype=raw_kv.dtype,
            )
            output_valid = torch.empty(
                batch_size,
                compressed_length,
                device=raw_kv.device,
                dtype=torch.bool,
            )
            block_dim = triton.next_power_of_2(kv_dim)
            if block_dim > 2048:
                raise ValueError("compressor Triton prototype supports KVDim <= 2048")
            forward_kernel[(batch_size * compressed_length,)](
                raw_kv,
                raw_gate,
                state_valid,
                output,
                output_valid,
                sequence_length=sequence_length,
                compressed_length=compressed_length,
                kv_dim=kv_dim,
                ratio=ratio,
                block_dim=block_dim,
            )
            ctx.save_for_backward(raw_kv, raw_gate, state_valid, output)
            ctx.ratio = ratio
            return output, output_valid

        @staticmethod
        def backward(
            ctx: Any, grad_output: Tensor, grad_output_valid: Tensor | None
        ) -> tuple[Tensor, Tensor, None, None]:
            del grad_output_valid
            raw_kv, raw_gate, state_valid, output = ctx.saved_tensors
            batch_size, sequence_length, _, kv_dim = raw_kv.shape
            compressed_length = output.shape[1]
            grad_raw_kv = torch.empty_like(raw_kv)
            grad_raw_gate = torch.empty_like(raw_gate)
            block_dim = triton.next_power_of_2(kv_dim)
            backward_kernel[(batch_size * sequence_length * 2,)](
                grad_output.contiguous(),
                raw_kv,
                raw_gate,
                state_valid,
                output,
                grad_raw_kv,
                grad_raw_gate,
                sequence_length=sequence_length,
                compressed_length=compressed_length,
                kv_dim=kv_dim,
                ratio=ctx.ratio,
                block_dim=block_dim,
            )
            return grad_raw_kv, grad_raw_gate, None, None

    def apply_pool(
        raw_kv: Tensor,
        raw_gate: Tensor,
        state_valid: Tensor,
        ratio: int,
    ) -> tuple[Tensor, Tensor]:
        result = _TritonCompressorPool.apply(raw_kv, raw_gate, state_valid, ratio)
        return cast(tuple[Tensor, Tensor], result)

    return apply_pool
