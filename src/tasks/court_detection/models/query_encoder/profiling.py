"""Deterministic capacity and runtime profiling for Court query models."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor, nn

from src.tasks.court_detection.model_io.adapters import CourtQueryModelIOAdapter
from src.tasks.court_detection.models.query_encoder.contracts import CourtEncoderTap
from src.tasks.court_detection.models.query_encoder.model import CourtQueryEncoderModel

JsonValue: TypeAlias = Any

PROFILE_SCHEMA = "court_query_profile_v1"
DECODER_MAC_DEFINITION = (
    "One multiply-accumulate is counted for every executed Conv2d output element "
    "times kernel_h*kernel_w*in_channels/groups and every executed Linear output "
    "element times in_features. Interpolation, normalization, activation, additions, "
    "and dense heads are excluded."
)


@dataclass(frozen=True, slots=True)
class QueryProfileInputContract:
    """One fixed tensor/device contract shared by every profile candidate."""

    batch_size: int
    channels: int
    height: int
    width: int
    dtype: str
    device: str

    def __post_init__(self) -> None:
        if min(self.batch_size, self.channels, self.height, self.width) <= 0:
            raise ValueError("Profile tensor dimensions must be positive.")
        if self.channels != 3:
            raise ValueError("Court query profiling requires three RGB channels.")
        if self.dtype != "float32":
            raise ValueError("Court query profiling currently fixes dtype=float32.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Profile device must be cpu or cuda.")

    def as_dict(self) -> dict[str, JsonValue]:
        return {
            "batch_size": self.batch_size,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "dtype": self.dtype,
            "device": self.device,
        }


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count unique tensor elements owned by a module."""
    parameters = {
        id(parameter): parameter
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    }
    return sum(parameter.numel() for parameter in parameters.values())


def count_decoder_macs(
    decoder: nn.Module,
    taps: tuple[CourtEncoderTap, ...],
    *,
    output_hw: tuple[int, int],
) -> int:
    """Count decoder Conv2d/Linear MACs under the frozen explicit definition."""
    macs = 0

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        nonlocal macs
        _ = inputs
        if not isinstance(output, Tensor):
            raise TypeError("Profiled Conv2d/Linear modules must return one Tensor.")
        if isinstance(module, nn.Conv2d):
            kernel_h, kernel_w = module.kernel_size
            macs += (
                output.numel()
                * kernel_h
                * kernel_w
                * module.in_channels
                // module.groups
            )
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    handles = [
        layer.register_forward_hook(hook)
        for layer in decoder.modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    ]
    try:
        with torch.no_grad():
            cast(Callable[..., Tensor], decoder)(taps, output_hw=output_hw)
    finally:
        for handle in handles:
            handle.remove()
    if macs <= 0:
        raise ValueError("Decoder MAC profile must execute at least one counted layer.")
    return macs


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_ms(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("Profile warmup must be non-negative and repeats positive.")
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        _synchronize(device)
        samples: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            operation()
            _synchronize(device)
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return statistics.fmean(samples), statistics.pstdev(samples)


def _peak_memory_bytes(
    operation: Callable[[], object],
    *,
    device: torch.device,
) -> tuple[int | None, str]:
    if device.type != "cuda":
        with torch.no_grad():
            operation()
        return None, "unavailable_for_cpu_diagnostic"
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        operation()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device)), "cuda.max_memory_allocated"


def profile_query_model(
    model: CourtQueryEncoderModel,
    adapter: CourtQueryModelIOAdapter,
    images: Tensor,
    *,
    family: str,
    size: str,
    warmup: int,
    repeats: int,
) -> dict[str, JsonValue]:
    """Profile one complete query model and its decoder under one fixed input."""
    if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.float32:
        raise ValueError("Profile images must be float32 with shape (B,3,H,W).")
    device = images.device
    model.eval()
    query_call = adapter.prepare_images(images)
    encoded = model.task_encoder(query_call.patch_batch)
    output_hw = (query_call.height, query_call.width)

    decoder_macs = count_decoder_macs(model.decoder, encoded.taps, output_hw=output_hw)

    def decoder_operation() -> object:
        return model.decoder(encoded.taps, output_hw=output_hw)

    def end_to_end_operation() -> object:
        prepared = adapter.prepare_images(images)
        return model(*prepared.model_args)

    decoder_mean, decoder_std = _latency_ms(
        decoder_operation,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    end_to_end_mean, end_to_end_std = _latency_ms(
        end_to_end_operation,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    peak_memory, memory_method = _peak_memory_bytes(
        end_to_end_operation,
        device=device,
    )
    evidence_kind = "gpu_runtime" if device.type == "cuda" else "cpu_diagnostic"
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    contract = QueryProfileInputContract(
        batch_size=int(images.shape[0]),
        channels=int(images.shape[1]),
        height=int(images.shape[2]),
        width=int(images.shape[3]),
        dtype="float32",
        device=device.type,
    )
    return {
        "schema": PROFILE_SCHEMA,
        "candidate": {"family": family, "size": size},
        "evidence": {
            "kind": evidence_kind,
            "device_name": device_name,
            "latency_is_adoption_evidence": evidence_kind == "gpu_runtime",
        },
        "execution_contract": {
            "model_mode": "eval",
            "autograd_enabled": False,
            "latency_statistic": "arithmetic_mean_and_population_std_ms",
            "peak_scope": "end_to_end_forward",
        },
        "input_contract": contract.as_dict(),
        "parameters": {
            "decoder": count_parameters(model.decoder),
            "trainable": count_parameters(model, trainable_only=True),
            "total": count_parameters(model),
        },
        "decoder_macs": {
            "count": decoder_macs,
            "definition": DECODER_MAC_DEFINITION,
        },
        "latency_ms": {
            "warmup": warmup,
            "repeats": repeats,
            "decoder_mean": decoder_mean,
            "decoder_std": decoder_std,
            "end_to_end_mean": end_to_end_mean,
            "end_to_end_std": end_to_end_std,
        },
        "peak_memory": {
            "bytes": peak_memory,
            "method": memory_method,
        },
    }


def validate_profile_record(
    value: Mapping[str, object],
    *,
    require_gpu_evidence: bool,
) -> None:
    """Fail closed when a profiler record is incomplete or mislabeled."""
    expected = {
        "schema",
        "candidate",
        "evidence",
        "execution_contract",
        "input_contract",
        "parameters",
        "decoder_macs",
        "latency_ms",
        "peak_memory",
    }
    if set(value) != expected or value["schema"] != PROFILE_SCHEMA:
        raise ValueError("Query profile record fields/schema changed.")
    candidate = _mapping(value["candidate"], name="profile.candidate")
    if set(candidate) != {"family", "size"} or any(
        not isinstance(candidate[name], str) or not candidate[name]
        for name in ("family", "size")
    ):
        raise ValueError("Query profile candidate identity is invalid.")
    if candidate["family"] not in {"linear", "progressive", "dpt"} or candidate[
        "size"
    ] not in {"tiny", "small", "base"}:
        raise ValueError("Query profile candidate family/size is unsupported.")
    evidence = _mapping(value["evidence"], name="profile.evidence")
    if set(evidence) != {"kind", "device_name", "latency_is_adoption_evidence"}:
        raise ValueError("Query profile evidence fields changed.")
    kind = evidence["kind"]
    if kind not in {"gpu_runtime", "cpu_diagnostic"}:
        raise ValueError("Query profile evidence kind is invalid.")
    if evidence["latency_is_adoption_evidence"] is not (kind == "gpu_runtime"):
        raise ValueError("Query profile latency evidence label is inconsistent.")
    if require_gpu_evidence and kind != "gpu_runtime":
        raise ValueError("Adoption summary requires GPU runtime profile evidence.")
    execution = _mapping(value["execution_contract"], name="profile.execution_contract")
    if execution != {
        "model_mode": "eval",
        "autograd_enabled": False,
        "latency_statistic": "arithmetic_mean_and_population_std_ms",
        "peak_scope": "end_to_end_forward",
    }:
        raise ValueError("Query profile execution contract changed.")

    contract = _mapping(value["input_contract"], name="profile.input_contract")
    if set(contract) != {
        "batch_size",
        "channels",
        "height",
        "width",
        "dtype",
        "device",
    }:
        raise ValueError("Query profile input contract fields changed.")
    QueryProfileInputContract(
        batch_size=_integer(contract["batch_size"], name="profile.batch_size"),
        channels=_integer(contract["channels"], name="profile.channels"),
        height=_integer(contract["height"], name="profile.height"),
        width=_integer(contract["width"], name="profile.width"),
        dtype=_string(contract["dtype"], name="profile.dtype"),
        device=_string(contract["device"], name="profile.device"),
    )
    parameters = _mapping(value["parameters"], name="profile.parameters")
    if set(parameters) != {"decoder", "trainable", "total"}:
        raise ValueError("Query profile parameter fields changed.")
    decoder = _positive_integer(parameters["decoder"], name="profile.decoder_params")
    trainable = _positive_integer(
        parameters["trainable"], name="profile.trainable_params"
    )
    total = _positive_integer(parameters["total"], name="profile.total_params")
    if decoder > total or trainable > total:
        raise ValueError("Query profile parameter counts are inconsistent.")
    macs = _mapping(value["decoder_macs"], name="profile.decoder_macs")
    if set(macs) != {"count", "definition"}:
        raise ValueError("Query profile decoder MAC fields changed.")
    _positive_integer(macs["count"], name="profile.decoder_macs.count")
    if macs["definition"] != DECODER_MAC_DEFINITION:
        raise ValueError("Query profile decoder MAC definition changed.")
    latency = _mapping(value["latency_ms"], name="profile.latency_ms")
    if set(latency) != {
        "warmup",
        "repeats",
        "decoder_mean",
        "decoder_std",
        "end_to_end_mean",
        "end_to_end_std",
    }:
        raise ValueError("Query profile latency fields changed.")
    _nonnegative_integer(latency["warmup"], name="profile.warmup")
    _positive_integer(latency["repeats"], name="profile.repeats")
    for name in ("decoder_mean", "decoder_std", "end_to_end_mean", "end_to_end_std"):
        number = _finite_number(latency[name], name=f"profile.{name}")
        if number < 0.0 or (name.endswith("mean") and number == 0.0):
            raise ValueError(f"profile.{name} must be positive/non-negative.")
    peak = _mapping(value["peak_memory"], name="profile.peak_memory")
    if set(peak) != {"bytes", "method"}:
        raise ValueError("Query profile peak-memory fields changed.")
    if kind == "gpu_runtime":
        _positive_integer(peak["bytes"], name="profile.peak_memory.bytes")
        if peak["method"] != "cuda.max_memory_allocated":
            raise ValueError("GPU peak memory must use cuda.max_memory_allocated.")
    elif (
        peak["bytes"] is not None or peak["method"] != "unavailable_for_cpu_diagnostic"
    ):
        raise ValueError("CPU diagnostics must not claim peak tensor-memory evidence.")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer <= 0:
        raise ValueError(f"{name} must be positive.")
    return integer


def _nonnegative_integer(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{name} must be numeric.")
    number = float(cast(float | int, value))
    if not torch.isfinite(torch.tensor(number)):
        raise ValueError(f"{name} must be finite.")
    return number


__all__ = [
    "DECODER_MAC_DEFINITION",
    "PROFILE_SCHEMA",
    "QueryProfileInputContract",
    "count_decoder_macs",
    "count_parameters",
    "profile_query_model",
    "validate_profile_record",
]
