"""Benchmark integrated BLCS track-query stages across temporal schedules.

The benchmark compares an all-global temporal baseline (A), the production
hybrid ``C,C,C,G`` schedule with the reference compressed-window executor (B),
and the same hybrid schedule with the explicit CUDA executor (C).  Every worker
runs the real mHC -> object temporal -> spatial -> query temporal stage stack;
this is intentionally not a standalone attention-kernel microbenchmark.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.benchmarks.contracts import (
    ISSUE_NUMBER,
    BenchmarkContractError,
    build_cuda_environment,
    build_source_record,
    load_json_object,
    repository_root,
    utc_timestamp,
    write_json_atomic,
)
from src.tasks.blcs.configuration import (
    TrackQueryCSWAConfig,
    TrackQueryMHCConfig,
    TrackQueryModelConfig,
)
from src.tasks.blcs.models.blcs_track_query_model import BLCSTrackQueryModel

COMPONENT = "integrated"
FROZEN_BASE_REVISION = "bc4577375e7ab14fbac6363f75b60e53add8121d"
SOURCE_FILES = (
    "src/tasks/blcs/benchmarks/track_query_integrated/__init__.py",
    "/".join(("src", "tasks", "blcs", "configs", "data", "_multiview.yaml")),
    "/".join(("src", "tasks", "blcs", "configs", "data", "tracking.yaml")),
    "src/tasks/blcs/configs/data/tracking_chunked.yaml",
    "src/tasks/blcs/configs/generation/multi_object.yaml",
    "src/tasks/blcs/configs/model/_track_query.yaml",
    "src/tasks/blcs/configs/model/track_query_base.yaml",
    "src/tasks/blcs/configs/model/track_query_large.yaml",
    "src/tasks/blcs/configs/model/track_query_small.yaml",
    "src/tasks/blcs/configs/train_tracking.yaml",
    "src/tasks/blcs/configs/train_tracking_chunked.yaml",
    "src/tasks/blcs/configuration.py",
    "src/tasks/blcs/models/blcs_track_query_model.py",
    "src/tasks/blcs/models/components/track_query_stage.py",
    "src/utils/models/components/block.py",
    "src/utils/models/components/compressor.py",
    "src/utils/models/components/cswa.py",
    "src/utils/models/components/mhc.py",
    "src/utils/models/components/ops/compressed_time_local/_autograd.py",
    "src/utils/models/components/ops/compressed_time_local/api.py",
    "src/utils/models/components/ops/compressed_time_local/bindings.cpp",
    "src/utils/models/components/ops/compressed_time_local/kernels.cu",
    "src/utils/models/components/ops/compressed_time_local/reference.py",
)
SEED = ISSUE_NUMBER
MASK_DENSITY = 0.875
WARMUP = 1
ITERATIONS = 3
DTYPES = ("float32", "bfloat16")
MEASUREMENTS = ("forward", "forward-backward")
CANDIDATES = ("A-global-only", "B-hybrid-reference", "C-hybrid-cuda")
RUN_STATUSES = frozenset({"ok", "oom", "unsupported", "unavailable"})
PARITY_STATUSES = frozenset({"pass", "fail", "not-applicable", "not-run"})
MEMORY_MEASUREMENT = {
    "scope": "process_local_cuda_allocator",
    "metrics": ["peak_allocated_bytes", "peak_reserved_bytes"],
    "capacity_invariant": (
        "0 <= peak_allocated_bytes <= peak_reserved_bytes <= "
        "environment.device_total_memory_bytes"
    ),
    "over_capacity_policy": "unsupported_without_physical_peak_claim",
}
TOLERANCES: dict[str, dict[str, float]] = {
    "float32": {"atol": 2.0e-5, "rtol": 2.0e-4},
    "bfloat16": {"atol": 6.5e-2, "rtol": 2.0e-2},
}


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Repository model width used by one benchmark case."""

    name: str
    source_config: str
    hidden_dim: int
    num_heads: int
    num_stages: int
    ffn_dim: int
    num_queries: int
    rope_dim: int

    @property
    def record(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "source_config": self.source_config,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_stages": self.num_stages,
            "ffn_dim": self.ffn_dim,
            "num_queries": self.num_queries,
            "rope_dim": self.rope_dim,
        }


MODEL_PROFILES = (
    ModelProfile(
        name="small",
        source_config="src/tasks/blcs/configs/model/track_query_small.yaml",
        hidden_dim=256,
        num_heads=4,
        num_stages=8,
        ffn_dim=704,
        num_queries=4,
        rope_dim=64,
    ),
    ModelProfile(
        name="base",
        source_config="src/tasks/blcs/configs/model/track_query_base.yaml",
        hidden_dim=512,
        num_heads=8,
        num_stages=8,
        ffn_dim=1408,
        num_queries=4,
        rope_dim=64,
    ),
    ModelProfile(
        name="large",
        source_config="src/tasks/blcs/configs/model/track_query_large.yaml",
        hidden_dim=512,
        num_heads=8,
        num_stages=12,
        ffn_dim=1408,
        num_queries=4,
        rope_dim=64,
    ),
)
_PROFILES_BY_NAME = {profile.name: profile for profile in MODEL_PROFILES}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One real-width stage shape and its repository-shape provenance."""

    name: str
    model_profile: str
    batch_size: int
    num_views: int
    frames: int
    provenance: str

    @property
    def profile(self) -> ModelProfile:
        return _PROFILES_BY_NAME[self.model_profile]

    @property
    def shape(self) -> dict[str, int | str]:
        profile = self.profile
        return {
            "batch_size": self.batch_size,
            "num_views": self.num_views,
            "frames": self.frames,
            "num_queries": profile.num_queries,
            "hidden_dim": profile.hidden_dim,
            "num_heads": profile.num_heads,
            "num_stages": profile.num_stages,
            "ffn_dim": profile.ffn_dim,
            "object_path_n": self.batch_size * self.num_views,
            "query_path_n": self.batch_size * profile.num_queries,
            "spatial_tokens": profile.num_queries
            + self.num_views * profile.num_queries,
            "provenance": self.provenance,
        }


REQUIRED_TRAINING_CASE = "configured-training-small-max-t1024"
REQUIRED_TRAINING_DTYPE = "bfloat16"
REQUIRED_TRAINING_MEASUREMENT = "forward-backward"

BENCHMARK_CASES = (
    BenchmarkCase(
        name="smoke-small",
        model_profile="small",
        batch_size=1,
        num_views=1,
        frames=32,
        provenance="correctness smoke with the repository small width",
    ),
    BenchmarkCase(
        name="historic-diagnostic-small-b8-t512",
        model_profile="small",
        batch_size=8,
        num_views=3,
        frames=512,
        provenance=(
            "historic pre-repair physical-B8/T512 diagnostic; not a configured "
            "production training shape"
        ),
    ),
    BenchmarkCase(
        name=REQUIRED_TRAINING_CASE,
        model_profile="small",
        batch_size=1,
        num_views=3,
        frames=1024,
        provenance=(
            "both tracking data configs physical batch=1; both tracking train "
            "roots accumulate_grad_batches=8; effective batch=8; view lower "
            "bound=3; maximum configured T=1024"
        ),
    ),
    BenchmarkCase(
        name="configured-inference-small-t1024",
        model_profile="small",
        batch_size=1,
        num_views=5,
        frames=1024,
        provenance="predictor batch=1, view upper bound=5, generated scene T=1024",
    ),
    BenchmarkCase(
        name="long-context-small-t2048",
        model_profile="small",
        batch_size=1,
        num_views=3,
        frames=2048,
        provenance="required long-context extension beyond configured T=1024",
    ),
    BenchmarkCase(
        name="configured-inference-base-t512",
        model_profile="base",
        batch_size=1,
        num_views=3,
        frames=512,
        provenance="repository base width at configured T/view lower bounds",
    ),
    BenchmarkCase(
        name="configured-inference-large-t512",
        model_profile="large",
        batch_size=1,
        num_views=3,
        frames=512,
        provenance="repository large width at configured T/view lower bounds",
    ),
)
_CASES_BY_NAME = {case.name: case for case in BENCHMARK_CASES}

PROTOCOL: dict[str, Any] = {
    "seed": SEED,
    "mask_density": MASK_DENSITY,
    "warmup": WARMUP,
    "iterations": ITERATIONS,
    "measurements": list(MEASUREMENTS),
    "dtypes": list(DTYPES),
    "production_mixed_precision_dtype": "bfloat16",
    "required_training_admission": {
        "case": REQUIRED_TRAINING_CASE,
        "model_profile": "small",
        "physical_batch_size": 1,
        "accumulate_grad_batches": 8,
        "effective_batch_size": 8,
        "num_views": 3,
        "frames": 1024,
        "configured_sequence_bound": "maximum",
        "dtype": REQUIRED_TRAINING_DTYPE,
        "measurement": REQUIRED_TRAINING_MEASUREMENT,
        "candidates": list(CANDIDATES),
    },
    "synchronize_each_iteration": True,
    "autocast": {"float32": False, "bfloat16": True},
    "compile": False,
    "dropout": 0.0,
    "module_mode": "eval",
    "scope": "full mHC/object-temporal/spatial/query-temporal stage stack",
    "throughput_definition": "input frames/s = batch_size * frames / median seconds",
    "memory_measurement": MEMORY_MEASUREMENT,
    "parity_scope": "stage outputs and camera/query input gradients",
    "model_profiles": [profile.record for profile in MODEL_PROFILES],
    "cases": [
        {"name": case.name, "model_profile": case.model_profile, **case.shape}
        for case in BENCHMARK_CASES
    ],
    "tolerances": TOLERANCES,
}

_WORKER_PREFIX = "INTEGRATED_WORKER_RESULT="
_Measurement = Literal["forward", "forward-backward"]


@dataclass(slots=True)
class _StageInputs:
    camera_tokens: Tensor
    slots: Tensor
    camera_state_valid: Tensor
    frame_mask: Tensor
    spatial_attention_mask: Tensor
    object_temporal_state_valid: Tensor
    object_temporal_attention_mask: Tensor
    query_temporal_state_valid: Tensor
    query_temporal_attention_mask: Tensor
    spatial_freqs: Tensor
    time_freqs: Tensor


class _PhysicalMemoryMeasurementUnsupported(RuntimeError):
    """Raised when allocator counters cannot be accepted as physical peaks."""


class _IntegratedStageStack(nn.Module):
    """Benchmark wrapper over the production track-query stage list."""

    def __init__(self, model: BLCSTrackQueryModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: _StageInputs) -> tuple[Tensor, Tensor]:
        camera_tokens = inputs.camera_tokens
        slots = inputs.slots
        for stage in self.model.stages:
            camera_tokens, slots = stage(
                camera_tokens,
                slots,
                camera_state_valid=inputs.camera_state_valid,
                frame_mask=inputs.frame_mask,
                spatial_attention_mask=inputs.spatial_attention_mask,
                object_temporal_state_valid=inputs.object_temporal_state_valid,
                object_temporal_attention_mask=(
                    inputs.object_temporal_attention_mask
                ),
                query_temporal_state_valid=inputs.query_temporal_state_valid,
                query_temporal_attention_mask=inputs.query_temporal_attention_mask,
                spatial_freqs=inputs.spatial_freqs,
                time_freqs=inputs.time_freqs,
            )
        return camera_tokens, slots


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported benchmark dtype: {dtype_name}")


def _model_config(
    profile: ModelProfile,
    *,
    backend: Literal["reference", "cuda"],
) -> TrackQueryModelConfig:
    return TrackQueryModelConfig(
        name="blcs_track_query",
        hidden_dim=profile.hidden_dim,
        num_heads=profile.num_heads,
        num_stages=profile.num_stages,
        ffn_dim=profile.ffn_dim,
        num_queries=profile.num_queries,
        rope_dim=profile.rope_dim,
        dropout=0.0,
        role_rope_enabled=True,
        mask_invisible_observations=True,
        invisible_init_std=0.02,
        observation_fusion="linear",
        point_fusion=None,
        mhc=TrackQueryMHCConfig(
            coefficient_dim=64,
            sinkhorn_iters=20,
            eps=1.0e-6,
            residual_identity_bias=4.0,
            update_scale_init=0.0,
        ),
        cswa=TrackQueryCSWAConfig(
            compression_ratio=4,
            window_radius=4,
            backend=backend,
        ),
    )


def _candidate_seed(case: BenchmarkCase, dtype_name: str) -> int:
    case_index = BENCHMARK_CASES.index(case)
    dtype_index = DTYPES.index(dtype_name)
    return int(SEED + 100 * case_index + dtype_index)


def _build_module(
    candidate: str,
    case: BenchmarkCase,
    dtype_name: str,
) -> _IntegratedStageStack:
    backend: Literal["reference", "cuda"] = (
        "cuda" if candidate == "C-hybrid-cuda" else "reference"
    )
    config = _model_config(case.profile, backend=backend)
    torch.manual_seed(_candidate_seed(case, dtype_name))
    model = BLCSTrackQueryModel(config)
    if candidate == "A-global-only":
        # Use the production stage class and model block builder, but assign
        # every logical stage a global index.  The baseline therefore retains
        # mHC, spatial work, widths, masks, and stage count while changing only
        # the temporal schedule under comparison.
        head_dim = case.profile.hidden_dim // case.profile.num_heads
        model.stages = nn.ModuleList(
            [
                model._build_stage(  # noqa: SLF001 - benchmark construction seam
                    stage_index=4 * logical_index + 3,
                    config=config,
                    head_dim=head_dim,
                )
                for logical_index in range(case.profile.num_stages)
            ]
        )
    elif candidate not in {"B-hybrid-reference", "C-hybrid-cuda"}:
        raise ValueError(f"unsupported integrated candidate: {candidate}")
    model = model.cuda()
    model.eval()
    return _IntegratedStageStack(model)


def _dense_self_attention_mask(state_valid: Tensor) -> Tensor:
    return state_valid.unsqueeze(-1) & state_valid.unsqueeze(-2)


def _make_inputs(
    module: _IntegratedStageStack,
    case: BenchmarkCase,
    dtype_name: str,
) -> _StageInputs:
    profile = case.profile
    device = torch.device("cuda")
    dtype = _torch_dtype(dtype_name)
    generator = torch.Generator(device=device)
    generator.manual_seed(_candidate_seed(case, dtype_name) + 1)
    camera_tokens = torch.randn(
        case.batch_size,
        case.num_views,
        case.frames,
        profile.num_queries,
        profile.hidden_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    slots = torch.randn(
        case.batch_size,
        case.frames,
        profile.num_queries,
        profile.hidden_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    frame_mask = (
        torch.rand(
            case.batch_size,
            case.frames,
            device=device,
            generator=generator,
        )
        < MASK_DENSITY
    )
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True
    camera_state_valid = (
        torch.rand(
            case.batch_size,
            case.num_views,
            case.frames,
            profile.num_queries,
            device=device,
            generator=generator,
        )
        < MASK_DENSITY
    ) & frame_mask[:, None, :, None]
    # Keep at least one object stream valid at every valid frame.  This makes
    # the fixed density reproducible while satisfying the CSWA local-key
    # contract for both the object and query paths.
    camera_state_valid[..., 0] |= frame_mask[:, None, :]
    camera_tokens = camera_tokens * camera_state_valid.unsqueeze(-1)
    slots = slots * frame_mask[:, :, None, None]

    object_state_valid = camera_state_valid.any(dim=-1).reshape(
        case.batch_size * case.num_views, case.frames
    )
    query_state_valid = (
        frame_mask[:, None, :]
        .expand(-1, profile.num_queries, -1)
        .reshape(case.batch_size * profile.num_queries, case.frames)
    )
    time_major_camera = camera_state_valid.permute(0, 2, 1, 3).reshape(
        case.batch_size, case.frames, -1
    )
    spatial_state_valid = torch.cat(
        (
            frame_mask[:, :, None].expand(-1, -1, profile.num_queries),
            time_major_camera,
        ),
        dim=-1,
    ).flatten(0, 1)
    spatial_mask = _dense_self_attention_mask(spatial_state_valid)
    object_mask = _dense_self_attention_mask(object_state_valid)
    query_mask = _dense_self_attention_mask(query_state_valid)

    coordinates = BLCSTrackQueryModel.build_spatial_coordinates(
        batch_size=case.batch_size,
        num_frames=case.frames,
        num_views=case.num_views,
        num_detections=profile.num_queries,
        num_queries=profile.num_queries,
        device=device,
    )
    spatial_freqs = module.model.spatial_frequency_computer(coordinates)
    time_freqs = module.model.temporal_frequency_computer(
        torch.arange(case.frames, device=device).unsqueeze(-1)
    )
    return _StageInputs(
        camera_tokens=camera_tokens,
        slots=slots,
        camera_state_valid=camera_state_valid,
        frame_mask=frame_mask,
        spatial_attention_mask=spatial_mask,
        object_temporal_state_valid=object_state_valid,
        object_temporal_attention_mask=object_mask,
        query_temporal_state_valid=query_state_valid,
        query_temporal_attention_mask=query_mask,
        spatial_freqs=spatial_freqs,
        time_freqs=time_freqs,
    )


def _autocast(dtype_name: str) -> torch.autocast:
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=dtype_name == "bfloat16",
    )


def _clear_gradients(module: nn.Module, inputs: _StageInputs) -> None:
    module.zero_grad(set_to_none=True)
    inputs.camera_tokens.grad = None
    inputs.slots.grad = None


def _run_iteration(
    module: _IntegratedStageStack,
    inputs: _StageInputs,
    measurement: _Measurement,
    dtype_name: str,
) -> tuple[Tensor, Tensor]:
    if measurement == "forward":
        with torch.no_grad(), _autocast(dtype_name):
            return cast(tuple[Tensor, Tensor], module(inputs))
    _clear_gradients(module, inputs)
    inputs.camera_tokens.requires_grad_(True)
    inputs.slots.requires_grad_(True)
    with _autocast(dtype_name):
        camera_output, slot_output = module(inputs)
        loss = camera_output.float().square().mean() + slot_output.float().square().mean()
    loss.backward()
    return camera_output, slot_output


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return float(sorted_values[index])


def _measure_candidate(
    module: _IntegratedStageStack,
    inputs: _StageInputs,
    measurement: _Measurement,
    dtype_name: str,
    case: BenchmarkCase,
    device_total_memory_bytes: int,
) -> tuple[dict[str, float], dict[str, int], dict[str, float | str]]:
    for _ in range(WARMUP):
        _run_iteration(module, inputs, measurement, dtype_name)
    torch.cuda.synchronize()

    _clear_gradients(module, inputs)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _run_iteration(module, inputs, measurement, dtype_name)
    torch.cuda.synchronize()
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    violation = _physical_memory_violation(memory, device_total_memory_bytes)
    if violation is not None:
        raise _PhysicalMemoryMeasurementUnsupported(violation)

    elapsed_ms: list[float] = []
    for _ in range(ITERATIONS):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _run_iteration(module, inputs, measurement, dtype_name)
        torch.cuda.synchronize()
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
    elapsed_ms.sort()
    median = _percentile(elapsed_ms, 0.5)
    p95 = _percentile(elapsed_ms, 0.95)
    frames = case.batch_size * case.frames
    return (
        {"median_ms": median, "p95_ms": p95},
        memory,
        {"unit": "input-frames/s", "value": frames / (median / 1000.0)},
    )


def _capture_parity_side(
    candidate: str,
    case: BenchmarkCase,
    dtype_name: str,
) -> tuple[list[Tensor], list[Tensor]]:
    module = _build_module(candidate, case, dtype_name)
    inputs = _make_inputs(module, case, dtype_name)
    _, _ = _run_iteration(
        module,
        inputs,
        "forward-backward",
        dtype_name,
    )
    with torch.no_grad(), _autocast(dtype_name):
        camera_output, slot_output = module(inputs)
    if inputs.camera_tokens.grad is None or inputs.slots.grad is None:
        raise RuntimeError("integrated parity inputs did not receive gradients")
    outputs = [camera_output.detach().float().cpu(), slot_output.detach().float().cpu()]
    gradients = [
        inputs.camera_tokens.grad.detach().float().cpu(),
        inputs.slots.grad.detach().float().cpu(),
    ]
    del inputs, module, camera_output, slot_output
    torch.cuda.empty_cache()
    return outputs, gradients


def _error_summary(
    reference: Sequence[Tensor], candidate: Sequence[Tensor]
) -> tuple[float, float]:
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for reference_tensor, candidate_tensor in zip(
        reference, candidate, strict=True
    ):
        difference = (reference_tensor - candidate_tensor).abs()
        maximum = max(maximum, float(difference.max().item()))
        absolute_sum += float(difference.sum().item())
        element_count += difference.numel()
    return maximum, absolute_sum / max(element_count, 1)


def _cuda_parity(case: BenchmarkCase, dtype_name: str) -> dict[str, Any]:
    reference_outputs, reference_gradients = _capture_parity_side(
        "B-hybrid-reference", case, dtype_name
    )
    candidate_outputs, candidate_gradients = _capture_parity_side(
        "C-hybrid-cuda", case, dtype_name
    )
    forward_max, forward_mean = _error_summary(
        reference_outputs, candidate_outputs
    )
    backward_max, backward_mean = _error_summary(
        reference_gradients, candidate_gradients
    )
    tolerance = TOLERANCES[dtype_name]
    forward_close = all(
        torch.allclose(
            reference,
            candidate,
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        )
        for reference, candidate in zip(
            reference_outputs, candidate_outputs, strict=True
        )
    )
    backward_close = all(
        torch.allclose(
            reference,
            candidate,
            atol=tolerance["atol"],
            rtol=tolerance["rtol"],
        )
        for reference, candidate in zip(
            reference_gradients, candidate_gradients, strict=True
        )
    )
    return {
        "status": "pass" if forward_close and backward_close else "fail",
        "reference_candidate": "B-hybrid-reference",
        "scope": PROTOCOL["parity_scope"],
        "forward_max_abs_error": forward_max,
        "forward_mean_abs_error": forward_mean,
        "backward_max_abs_error": backward_max,
        "backward_mean_abs_error": backward_mean,
        "atol": tolerance["atol"],
        "rtol": tolerance["rtol"],
    }


def _self_parity(candidate: str, dtype_name: str) -> dict[str, Any]:
    if candidate == "A-global-only":
        return {
            "status": "not-applicable",
            "reference_candidate": None,
            "scope": PROTOCOL["parity_scope"],
            "forward_max_abs_error": None,
            "forward_mean_abs_error": None,
            "backward_max_abs_error": None,
            "backward_mean_abs_error": None,
            "atol": None,
            "rtol": None,
        }
    tolerance = TOLERANCES[dtype_name]
    return {
        "status": "pass",
        "reference_candidate": "self",
        "scope": PROTOCOL["parity_scope"],
        "forward_max_abs_error": 0.0,
        "forward_mean_abs_error": 0.0,
        "backward_max_abs_error": 0.0,
        "backward_mean_abs_error": 0.0,
        "atol": tolerance["atol"],
        "rtol": tolerance["rtol"],
    }


def _not_run_parity() -> dict[str, Any]:
    return {
        "status": "not-run",
        "reference_candidate": None,
        "scope": PROTOCOL["parity_scope"],
        "forward_max_abs_error": None,
        "forward_mean_abs_error": None,
        "backward_max_abs_error": None,
        "backward_mean_abs_error": None,
        "atol": None,
        "rtol": None,
    }


def _run_shape(case: BenchmarkCase, measurement: str) -> dict[str, int | str]:
    return {**case.shape, "measurement": measurement}


def _architecture(candidate: str) -> str:
    return "global-only" if candidate == "A-global-only" else "hybrid-CCCG"


def _backend(candidate: str) -> str:
    if candidate == "C-hybrid-cuda":
        return "cuda"
    if candidate == "B-hybrid-reference":
        return "reference"
    return "dense-pytorch"


def _available_run(
    *,
    case: BenchmarkCase,
    candidate: str,
    dtype_name: str,
    measurement: str,
    parameter_count: int,
    latency: Mapping[str, float],
    memory: Mapping[str, int],
    throughput: Mapping[str, float | str],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": case.name,
        "model_profile": case.model_profile,
        "candidate": candidate,
        "architecture": _architecture(candidate),
        "backend": _backend(candidate),
        "available": True,
        "status": "ok",
        "shape": _run_shape(case, measurement),
        "dtype": dtype_name,
        "measurement": measurement,
        "parameter_count": parameter_count,
        "policy": {
            "autocast": dtype_name == "bfloat16",
            "compile": False,
            "dropout": 0.0,
            "module_mode": "eval",
        },
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": dict(latency),
        "throughput": dict(throughput),
        "memory": dict(memory),
        "parity": dict(parity),
        "unavailable_reason": None,
    }


def _unavailable_run(
    *,
    case: BenchmarkCase,
    candidate: str,
    dtype_name: str,
    measurement: str,
    status: Literal["oom", "unsupported", "unavailable"],
    reason: str,
    parameter_count: int | None,
) -> dict[str, Any]:
    return {
        "case": case.name,
        "model_profile": case.model_profile,
        "candidate": candidate,
        "architecture": _architecture(candidate),
        "backend": _backend(candidate),
        "available": False,
        "status": status,
        "shape": _run_shape(case, measurement),
        "dtype": dtype_name,
        "measurement": measurement,
        "parameter_count": parameter_count,
        "policy": {
            "autocast": dtype_name == "bfloat16",
            "compile": False,
            "dropout": 0.0,
            "module_mode": "eval",
        },
        "warmup": WARMUP,
        "iterations": ITERATIONS,
        "latency": None,
        "throughput": None,
        "memory": None,
        "parity": _not_run_parity(),
        "unavailable_reason": reason,
    }


def _classify_unavailability(error: BaseException) -> tuple[str, str]:
    reason = f"{type(error).__name__}: {error}".replace("\n", " ")[:1000]
    lowered = reason.lower()
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in lowered:
        return "oom", reason
    unsupported_fragments = (
        "backend was requested",
        "extension is unavailable",
        "does not support",
        "unsupported",
    )
    if any(fragment in lowered for fragment in unsupported_fragments):
        return "unsupported", reason
    return "unavailable", reason


def _physical_memory_violation(
    memory: Mapping[str, object], device_total_memory_bytes: int
) -> str | None:
    """Return why allocator counters cannot represent a truthful physical peak."""
    if device_total_memory_bytes <= 0:
        return "recorded physical CUDA device capacity must be positive"
    allocated = memory.get("peak_allocated_bytes")
    reserved = memory.get("peak_reserved_bytes")
    if (
        isinstance(allocated, bool)
        or not isinstance(allocated, int)
        or allocated < 0
        or isinstance(reserved, bool)
        or not isinstance(reserved, int)
        or reserved < 0
    ):
        return "CUDA allocator peak counters must be non-negative integers"
    if allocated > reserved:
        return (
            "CUDA allocator peak_allocated_bytes "
            f"{allocated} exceeds peak_reserved_bytes {reserved}"
        )
    if allocated > device_total_memory_bytes or reserved > device_total_memory_bytes:
        return (
            "unsupported physical-memory measurement: process-local CUDA allocator "
            f"counters allocated={allocated}, reserved={reserved} exceed recorded "
            f"physical device capacity={device_total_memory_bytes}; no physical "
            "peak is claimed"
        )
    return None


def _run_worker(
    candidate: str,
    case_name: str,
    dtype_name: str,
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch CUDA is unavailable")
    case = _CASES_BY_NAME[case_name]
    device_total_memory_bytes = int(
        torch.cuda.get_device_properties(torch.device("cuda")).total_memory
    )
    module: _IntegratedStageStack | None = None
    parameter_count: int | None = None
    try:
        parity = (
            _cuda_parity(case, dtype_name)
            if candidate == "C-hybrid-cuda"
            else _self_parity(candidate, dtype_name)
        )
        if parity["status"] == "fail":
            raise RuntimeError(
                "integrated CUDA forward/backward parity failed for "
                f"{case_name}/{dtype_name}"
            )
        module = _build_module(candidate, case, dtype_name)
        parameter_count = sum(parameter.numel() for parameter in module.parameters())
        inputs = _make_inputs(module, case, dtype_name)
        runs: list[dict[str, Any]] = []
        for measurement in MEASUREMENTS:
            try:
                latency, memory, throughput = _measure_candidate(
                    module,
                    inputs,
                    cast(_Measurement, measurement),
                    dtype_name,
                    case,
                    device_total_memory_bytes,
                )
            except _PhysicalMemoryMeasurementUnsupported as error:
                runs.append(
                    _unavailable_run(
                        case=case,
                        candidate=candidate,
                        dtype_name=dtype_name,
                        measurement=measurement,
                        status="unsupported",
                        reason=str(error)[:1000],
                        parameter_count=parameter_count,
                    )
                )
                continue
            except torch.cuda.OutOfMemoryError as error:
                status, reason = _classify_unavailability(error)
                runs.append(
                    _unavailable_run(
                        case=case,
                        candidate=candidate,
                        dtype_name=dtype_name,
                        measurement=measurement,
                        status=cast(
                            Literal["oom", "unsupported", "unavailable"], status
                        ),
                        reason=reason,
                        parameter_count=parameter_count,
                    )
                )
                continue
            runs.append(
                _available_run(
                    case=case,
                    candidate=candidate,
                    dtype_name=dtype_name,
                    measurement=measurement,
                    parameter_count=parameter_count,
                    latency=latency,
                    memory=memory,
                    throughput=throughput,
                    parity=parity,
                )
            )
        return runs
    except torch.cuda.OutOfMemoryError as error:
        status, reason = _classify_unavailability(error)
        return [
            _unavailable_run(
                case=case,
                candidate=candidate,
                dtype_name=dtype_name,
                measurement=measurement,
                status=cast(Literal["oom", "unsupported", "unavailable"], status),
                reason=reason,
                parameter_count=parameter_count,
            )
            for measurement in MEASUREMENTS
        ]
    finally:
        del module
        torch.cuda.empty_cache()


def _profile_subprocess(
    candidate: str,
    case: BenchmarkCase,
    dtype_name: str,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "src.tasks.blcs.benchmarks.track_query_integrated",
        "--worker-candidate",
        candidate,
        "--worker-case",
        case.name,
        "--worker-dtype",
        dtype_name,
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
        lowered = reason.lower()
        if "out of memory" in lowered:
            status: Literal["oom", "unsupported", "unavailable"] = "oom"
        elif any(
            fragment in lowered
            for fragment in (
                "backend was requested",
                "extension is unavailable",
                "does not support",
                "unsupported",
            )
        ):
            status = "unsupported"
        else:
            raise RuntimeError(
                f"integrated worker failed for {case.name}/{dtype_name}/{candidate}: "
                f"{reason}"
            )
        return [
            _unavailable_run(
                case=case,
                candidate=candidate,
                dtype_name=dtype_name,
                measurement=measurement,
                status=status,
                reason=reason[:1000],
                parameter_count=None,
            )
            for measurement in MEASUREMENTS
        ]
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(_WORKER_PREFIX):
            payload = json.loads(line.removeprefix(_WORKER_PREFIX))
            if isinstance(payload, list) and all(
                isinstance(item, dict) for item in payload
            ):
                return cast(list[dict[str, Any]], payload)
            break
    raise RuntimeError(
        f"integrated worker {case.name}/{dtype_name}/{candidate} did not emit a result"
    )


def _comparison_key(run: Mapping[str, Any]) -> tuple[str, str, str]:
    return cast(str, run["case"]), cast(str, run["dtype"]), cast(
        str, run["measurement"]
    )


def _is_positive_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _required_training_run_failure(
    run: Mapping[str, Any],
    *,
    candidate: str,
    device_total_memory_bytes: int,
) -> str | None:
    """Return why one maximum configured-training admission row is invalid."""
    if run.get("available") is not True or run.get("status") != "ok":
        return f"{candidate} must be available=true,status=ok"

    latency = run.get("latency")
    if not isinstance(latency, Mapping) or any(
        not _is_positive_finite_number(latency.get(key))
        for key in ("median_ms", "p95_ms")
    ):
        return f"{candidate} must carry positive finite latency"

    throughput = run.get("throughput")
    if (
        not isinstance(throughput, Mapping)
        or throughput.get("unit") != "input-frames/s"
        or not _is_positive_finite_number(throughput.get("value"))
    ):
        return f"{candidate} must carry positive finite throughput"

    memory = run.get("memory")
    if not isinstance(memory, Mapping):
        return f"{candidate} must carry CUDA allocator memory"
    memory_violation = _physical_memory_violation(
        cast(Mapping[str, object], memory), device_total_memory_bytes
    )
    if memory_violation is not None:
        return f"{candidate} memory is not capacity-valid: {memory_violation}"

    parity = run.get("parity")
    expected_parity = "not-applicable" if candidate == "A-global-only" else "pass"
    if not isinstance(parity, Mapping) or parity.get("status") != expected_parity:
        return f"{candidate} parity must be {expected_parity}"
    return None


def decide_integrated_benchmark(
    runs: Sequence[Mapping[str, Any]],
    *,
    device_total_memory_bytes: int,
) -> dict[str, Any]:
    """Require smoke and maximum-training admission; record other omissions."""
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_comparison_key(run), {})[cast(str, run["candidate"])] = run

    comparisons: dict[str, dict[str, Any]] = {}
    complete_triplets = 0
    failed_reasons: list[str] = []
    for case in BENCHMARK_CASES:
        for dtype_name in DTYPES:
            for measurement in MEASUREMENTS:
                key = (case.name, dtype_name, measurement)
                case_runs = grouped.get(key, {})
                label = "::".join(key)
                if set(case_runs) != set(CANDIDATES):
                    failed_reasons.append(f"{label}: missing candidate record")
                    comparisons[label] = {
                        "complete": False,
                        "status": "missing",
                        "latency_ms": None,
                        "peak_allocated_bytes": None,
                        "speedup_B_vs_A": None,
                        "speedup_C_vs_A": None,
                        "speedup_C_vs_B": None,
                    }
                    continue
                available = all(
                    case_runs[candidate].get("available") is True
                    for candidate in CANDIDATES
                )
                if not available:
                    statuses = {
                        candidate: case_runs[candidate].get("status")
                        for candidate in CANDIDATES
                    }
                    comparisons[label] = {
                        "complete": False,
                        "status": statuses,
                        "latency_ms": None,
                        "peak_allocated_bytes": None,
                        "speedup_B_vs_A": None,
                        "speedup_C_vs_A": None,
                        "speedup_C_vs_B": None,
                    }
                    continue
                complete_triplets += 1
                latencies = {
                    candidate: float(
                        cast(Mapping[str, Any], case_runs[candidate]["latency"])[
                            "median_ms"
                        ]
                    )
                    for candidate in CANDIDATES
                }
                memory = {
                    candidate: int(
                        cast(Mapping[str, Any], case_runs[candidate]["memory"])[
                            "peak_allocated_bytes"
                        ]
                    )
                    for candidate in CANDIDATES
                }
                comparisons[label] = {
                    "complete": True,
                    "status": "measured",
                    "latency_ms": latencies,
                    "peak_allocated_bytes": memory,
                    "speedup_B_vs_A": latencies["A-global-only"]
                    / latencies["B-hybrid-reference"],
                    "speedup_C_vs_A": latencies["A-global-only"]
                    / latencies["C-hybrid-cuda"],
                    "speedup_C_vs_B": latencies["B-hybrid-reference"]
                    / latencies["C-hybrid-cuda"],
                }

    required_smoke = {
        ("smoke-small", dtype_name, measurement)
        for dtype_name in DTYPES
        for measurement in MEASUREMENTS
    }
    for key in required_smoke:
        candidate_runs = grouped.get(key, {})
        if set(candidate_runs) != set(CANDIDATES) or not all(
            candidate_runs[candidate]["available"] is True
            for candidate in CANDIDATES
        ):
            failed_reasons.append("::".join(key) + ": smoke triplet unavailable")

    required_training_key = (
        REQUIRED_TRAINING_CASE,
        REQUIRED_TRAINING_DTYPE,
        REQUIRED_TRAINING_MEASUREMENT,
    )
    required_training_runs = grouped.get(required_training_key, {})
    required_training_label = "::".join(required_training_key)
    if set(required_training_runs) != set(CANDIDATES):
        failed_reasons.append(
            required_training_label + ": required training candidate record missing"
        )
    else:
        for candidate in CANDIDATES:
            failure = _required_training_run_failure(
                required_training_runs[candidate],
                candidate=candidate,
                device_total_memory_bytes=device_total_memory_bytes,
            )
            if failure is not None:
                failed_reasons.append(f"{required_training_label}: {failure}")

    for run in runs:
        if run["available"] is True and run["candidate"] == "C-hybrid-cuda":
            parity = cast(Mapping[str, Any], run["parity"])
            if parity["status"] != "pass":
                failed_reasons.append(
                    "::".join(_comparison_key(run)) + ": CUDA parity failed"
                )

    return {
        "status": "PASS" if not failed_reasons else "FAIL",
        "reason": (
            "All required smoke and maximum configured-training A/B/C comparisons "
            "passed; every optional larger shape is measured or explicitly records "
            "OOM/unsupported status."
            if not failed_reasons
            else "; ".join(failed_reasons)
        ),
        "complete_same_shape_triplets": complete_triplets,
        "comparisons": comparisons,
    }


def _candidate_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    root = repository_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "frozen_base_revision": FROZEN_BASE_REVISION,
        "git_commit": source["git_commit"],
        "git_branch": branch,
        "component_source_sha256": source["fingerprint_sha256"],
        "scope": source["files"],
    }


def profile_integrated_benchmark() -> dict[str, Any]:
    """Run every A/B/C shape in isolated CUDA subprocesses."""
    environment = build_cuda_environment()
    properties = torch.cuda.get_device_properties(torch.device("cuda"))
    environment = {
        **environment,
        "device_total_memory_bytes": int(properties.total_memory),
        "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    runs = [
        run
        for case in BENCHMARK_CASES
        for dtype_name in DTYPES
        for candidate in CANDIDATES
        for run in _profile_subprocess(candidate, case, dtype_name)
    ]
    source = build_source_record(repository_root(), SOURCE_FILES)
    report = {
        "schema_version": 1,
        "issue": ISSUE_NUMBER,
        "component": COMPONENT,
        "generated_at_utc": utc_timestamp(),
        "source": source,
        "candidate_identity": _candidate_identity(source),
        "environment": environment,
        "protocol": PROTOCOL,
        "runs": runs,
        "decision": decide_integrated_benchmark(
            runs,
            device_total_memory_bytes=environment["device_total_memory_bytes"],
        ),
        "restrictions": [
            "Backends are construction-time explicit; C never falls back to B.",
            "CUDA CSWA supports float16, bfloat16, and float32, window_radius <= 64, and training attention dropout 0 only.",
            "The benchmark fixes dropout=0, compile=false, and eval mode for A/B/C; bfloat16 uses CUDA autocast with bfloat16 stage inputs.",
            "Unavailable and OOM records have no latency, throughput, memory, or parity claims.",
            "Allocator counters above recorded physical device capacity are classified as unsupported and never retained as successful physical peaks.",
        ],
        "risks": [
            "The global stage retained by the C,C,C,G schedule still has quadratic temporal masks and may OOM at long context.",
            "Integrated backward parity covers stage outputs and input gradients; component evidence separately covers CUDA Q/K/V gradients.",
            "Latency and memory are device-specific and use three measured iterations after one warmup to keep the full shape matrix reproducible.",
            "mHC and compressor remain PyTorch after their independent NO-GO gates, so C accelerates only compressed-window attention.",
        ],
    }
    validate_integrated_evidence(report)
    return report


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        raise BenchmarkContractError(
            f"{path} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise BenchmarkContractError(f"{path} must be an object with string keys")
    return cast(Mapping[str, Any], value)


def _require_nonempty_string(value: object, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"{path} must be a non-empty string")


def _require_finite_non_negative(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkContractError(f"{path} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise BenchmarkContractError(f"{path} must be a finite non-negative number")


def _validate_integrated_run(
    run: Mapping[str, Any], index: int, device_total_memory_bytes: int
) -> None:
    path = f"runs[{index}]"
    _require_exact_keys(
        run,
        {
            "case",
            "model_profile",
            "candidate",
            "architecture",
            "backend",
            "available",
            "status",
            "shape",
            "dtype",
            "measurement",
            "parameter_count",
            "policy",
            "warmup",
            "iterations",
            "latency",
            "throughput",
            "memory",
            "parity",
            "unavailable_reason",
        },
        path,
    )
    for key in (
        "case",
        "model_profile",
        "candidate",
        "architecture",
        "backend",
        "dtype",
        "measurement",
        "status",
    ):
        _require_nonempty_string(run[key], f"{path}.{key}")
    if run["candidate"] not in CANDIDATES:
        raise BenchmarkContractError(f"{path}.candidate is unsupported")
    if run["dtype"] not in DTYPES or run["measurement"] not in MEASUREMENTS:
        raise BenchmarkContractError(f"{path} dtype/measurement is unsupported")
    if run["status"] not in RUN_STATUSES:
        raise BenchmarkContractError(f"{path}.status is unsupported")
    if not isinstance(run["available"], bool):
        raise BenchmarkContractError(f"{path}.available must be boolean")
    case = _CASES_BY_NAME.get(cast(str, run["case"]))
    if case is None:
        raise BenchmarkContractError(f"{path}.case is unknown")
    if run["model_profile"] != case.model_profile:
        raise BenchmarkContractError(f"{path}.model_profile mismatch")
    if run["shape"] != _run_shape(case, cast(str, run["measurement"])):
        raise BenchmarkContractError(f"{path}.shape mismatch")
    if run["architecture"] != _architecture(cast(str, run["candidate"])):
        raise BenchmarkContractError(f"{path}.architecture mismatch")
    if run["backend"] != _backend(cast(str, run["candidate"])):
        raise BenchmarkContractError(f"{path}.backend mismatch")
    if run["warmup"] != WARMUP or run["iterations"] != ITERATIONS:
        raise BenchmarkContractError(f"{path} warmup/iterations mismatch")
    expected_policy = {
        "autocast": run["dtype"] == "bfloat16",
        "compile": False,
        "dropout": 0.0,
        "module_mode": "eval",
    }
    if run["policy"] != expected_policy:
        raise BenchmarkContractError(f"{path}.policy mismatch")

    parity = _require_mapping(run["parity"], f"{path}.parity")
    _require_exact_keys(
        parity,
        {
            "status",
            "reference_candidate",
            "scope",
            "forward_max_abs_error",
            "forward_mean_abs_error",
            "backward_max_abs_error",
            "backward_mean_abs_error",
            "atol",
            "rtol",
        },
        f"{path}.parity",
    )
    if parity["status"] not in PARITY_STATUSES:
        raise BenchmarkContractError(f"{path}.parity.status is unsupported")
    if parity["scope"] != PROTOCOL["parity_scope"]:
        raise BenchmarkContractError(f"{path}.parity.scope mismatch")

    if run["available"]:
        if run["status"] != "ok" or run["unavailable_reason"] is not None:
            raise BenchmarkContractError(f"{path} available status mismatch")
        if (
            isinstance(run["parameter_count"], bool)
            or not isinstance(run["parameter_count"], int)
            or run["parameter_count"] <= 0
        ):
            raise BenchmarkContractError(
                f"{path}.parameter_count must be a positive integer"
            )
        latency = _require_mapping(run["latency"], f"{path}.latency")
        _require_exact_keys(latency, {"median_ms", "p95_ms"}, f"{path}.latency")
        for key in latency:
            _require_finite_non_negative(latency[key], f"{path}.latency.{key}")
            if float(latency[key]) <= 0:
                raise BenchmarkContractError(f"{path}.latency.{key} must be positive")
        throughput = _require_mapping(run["throughput"], f"{path}.throughput")
        _require_exact_keys(throughput, {"unit", "value"}, f"{path}.throughput")
        if throughput["unit"] != "input-frames/s":
            raise BenchmarkContractError(f"{path}.throughput.unit mismatch")
        _require_finite_non_negative(throughput["value"], f"{path}.throughput.value")
        memory = _require_mapping(run["memory"], f"{path}.memory")
        _require_exact_keys(
            memory,
            {"peak_allocated_bytes", "peak_reserved_bytes"},
            f"{path}.memory",
        )
        for key in memory:
            if isinstance(memory[key], bool) or not isinstance(memory[key], int):
                raise BenchmarkContractError(f"{path}.memory.{key} must be integer")
            _require_finite_non_negative(memory[key], f"{path}.memory.{key}")
        violation = _physical_memory_violation(memory, device_total_memory_bytes)
        if violation is not None:
            raise BenchmarkContractError(f"{path}.memory violates capacity: {violation}")
        expected_parity_status = (
            "not-applicable" if run["candidate"] == "A-global-only" else "pass"
        )
        if parity["status"] != expected_parity_status:
            raise BenchmarkContractError(f"{path}.parity.status mismatch")
        if parity["status"] == "pass":
            for key in (
                "forward_max_abs_error",
                "forward_mean_abs_error",
                "backward_max_abs_error",
                "backward_mean_abs_error",
                "atol",
                "rtol",
            ):
                _require_finite_non_negative(parity[key], f"{path}.parity.{key}")
    else:
        if run["status"] not in {"oom", "unsupported", "unavailable"}:
            raise BenchmarkContractError(f"{path} unavailable status mismatch")
        _require_nonempty_string(run["unavailable_reason"], f"{path}.unavailable_reason")
        for key in ("latency", "throughput", "memory"):
            if run[key] is not None:
                raise BenchmarkContractError(f"{path}.{key} must be null")
        if parity["status"] != "not-run":
            raise BenchmarkContractError(f"{path}.parity must be not-run")


def validate_integrated_evidence(document: Mapping[str, Any]) -> None:
    """Validate the complete integrated schema and computed PASS decision."""
    _require_exact_keys(
        document,
        {
            "schema_version",
            "issue",
            "component",
            "generated_at_utc",
            "source",
            "candidate_identity",
            "environment",
            "protocol",
            "runs",
            "decision",
            "restrictions",
            "risks",
        },
        "evidence",
    )
    if document["schema_version"] != 1 or document["issue"] != ISSUE_NUMBER:
        raise BenchmarkContractError("integrated schema_version/issue mismatch")
    if document["component"] != COMPONENT:
        raise BenchmarkContractError("integrated component mismatch")
    _require_nonempty_string(document["generated_at_utc"], "generated_at_utc")

    source = _require_mapping(document["source"], "source")
    _require_exact_keys(
        source,
        {"git_commit", "files", "fingerprint_sha256"},
        "source",
    )
    expected_source = build_source_record(repository_root(), SOURCE_FILES)
    if source["files"] != expected_source["files"]:
        raise BenchmarkContractError("source.files mismatch")
    if source["fingerprint_sha256"] != expected_source["fingerprint_sha256"]:
        raise BenchmarkContractError("source.fingerprint_sha256 mismatch")
    _require_nonempty_string(source["git_commit"], "source.git_commit")

    identity = _require_mapping(document["candidate_identity"], "candidate_identity")
    _require_exact_keys(
        identity,
        {
            "frozen_base_revision",
            "git_commit",
            "git_branch",
            "component_source_sha256",
            "scope",
        },
        "candidate_identity",
    )
    if identity["frozen_base_revision"] != FROZEN_BASE_REVISION:
        raise BenchmarkContractError("candidate frozen base mismatch")
    if identity["git_commit"] != source["git_commit"]:
        raise BenchmarkContractError("candidate git commit mismatch")
    if identity["component_source_sha256"] != source["fingerprint_sha256"]:
        raise BenchmarkContractError("candidate source fingerprint mismatch")
    if identity["scope"] != source["files"]:
        raise BenchmarkContractError("candidate source scope mismatch")
    _require_nonempty_string(identity["git_branch"], "candidate_identity.git_branch")

    environment = _require_mapping(document["environment"], "environment")
    _require_exact_keys(
        environment,
        {
            "python",
            "torch",
            "cuda",
            "gpu",
            "compute_capability",
            "device_total_memory_bytes",
            "bfloat16_supported",
        },
        "environment",
    )
    for key in ("python", "torch", "cuda", "gpu", "compute_capability"):
        _require_nonempty_string(environment[key], f"environment.{key}")
    if (
        isinstance(environment["device_total_memory_bytes"], bool)
        or not isinstance(environment["device_total_memory_bytes"], int)
        or environment["device_total_memory_bytes"] <= 0
    ):
        raise BenchmarkContractError("environment.device_total_memory_bytes invalid")
    if environment["bfloat16_supported"] is not True:
        raise BenchmarkContractError("production bfloat16 must be supported")
    if document["protocol"] != PROTOCOL:
        raise BenchmarkContractError("integrated protocol mismatch")

    raw_runs = document["runs"]
    if not isinstance(raw_runs, list):
        raise BenchmarkContractError("runs must be an array")
    runs = cast(list[Mapping[str, Any]], raw_runs)
    expected_order = [
        (case.name, dtype_name, candidate, measurement)
        for case in BENCHMARK_CASES
        for dtype_name in DTYPES
        for candidate in CANDIDATES
        for measurement in MEASUREMENTS
    ]
    actual_order = [
        (run.get("case"), run.get("dtype"), run.get("candidate"), run.get("measurement"))
        for run in runs
    ]
    if actual_order != expected_order:
        raise BenchmarkContractError("integrated run matrix/order mismatch")
    device_total_memory_bytes = environment["device_total_memory_bytes"]
    for index, run in enumerate(runs):
        _validate_integrated_run(run, index, device_total_memory_bytes)

    expected_decision = decide_integrated_benchmark(
        runs,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if document["decision"] != expected_decision:
        raise BenchmarkContractError("integrated decision does not match runs")
    if cast(Mapping[str, Any], document["decision"])["status"] != "PASS":
        raise BenchmarkContractError("integrated benchmark decision must PASS")
    for key in ("restrictions", "risks"):
        values = document[key]
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise BenchmarkContractError(f"{key} must contain non-empty strings")


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
        "availability": {
            "::".join(
                (
                    cast(str, run["case"]),
                    cast(str, run["dtype"]),
                    cast(str, run["candidate"]),
                    cast(str, run["measurement"]),
                )
            ): (run["available"], run["status"])
            for run in runs
        },
        "parity": {
            "::".join(
                (
                    cast(str, run["case"]),
                    cast(str, run["dtype"]),
                    cast(str, run["candidate"]),
                    cast(str, run["measurement"]),
                )
            ): cast(Mapping[str, Any], run["parity"])["status"]
            for run in runs
        },
        "decision": cast(Mapping[str, Any], document["decision"])["status"],
    }


def execute(
    *,
    evidence_path: Path,
    runtime_result_path: Path,
    record_evidence: bool,
) -> dict[str, Any]:
    """Write fresh runtime output and record or validate stable evidence."""
    if evidence_path.resolve() == runtime_result_path.resolve():
        raise BenchmarkContractError("evidence and runtime-result paths must differ")
    runtime = profile_integrated_benchmark()
    write_json_atomic(runtime_result_path, runtime)
    if record_evidence:
        write_json_atomic(evidence_path, runtime)
        return runtime
    stable = load_json_object(evidence_path)
    validate_integrated_evidence(stable)
    if _semantic_signature(stable) != _semantic_signature(runtime):
        raise BenchmarkContractError(
            "fresh integrated availability/parity/decision differs from stable evidence"
        )
    return runtime
