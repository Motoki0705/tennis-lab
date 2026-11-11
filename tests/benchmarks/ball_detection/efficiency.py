"""Synchronized CUDA phases, raw samples and memory for untrained modules.

Callers own the fixed architecture/shape matrix. This harness never falls back
from CUDA/compile and never treats an uncounted custom operation as zero MACs.
"""

from __future__ import annotations

import math
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src.tasks.ball_detection.training.runtime.compilation import (
    compile_coordinate_model,
    coordinate_compilation_report,
    coordinate_compile_scope,
)


def quantiles(samples: Iterable[float]) -> dict[str, float]:
    values: NDArray[np.float64] = np.asarray(list(samples), dtype=np.float64)
    if not values.size or not bool(np.isfinite(values).all()):
        raise ValueError("Latency samples must be nonempty and finite")
    return dict(
        zip(
            ("p50", "p90", "p95"),
            (float(x) for x in np.percentile(values, (50, 90, 95))),
            strict=True,
        )
    )


def tensor_bytes(tree: Any) -> int:
    seen: set[tuple[str, int]] = set()

    def count(value: Any) -> int:
        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        if isinstance(value, dict):
            return sum(count(x) for x in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(x) for x in value)
        return 0

    return count(tree)


def environment() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA L4":
        raise RuntimeError("The specified L4 is required")
    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        precision="BF16 autocast; official MSDA FP32",
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        sdpa=dict(
            flash=torch.backends.cuda.flash_sdp_enabled(),
            memory_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            math=torch.backends.cuda.math_sdp_enabled(),
        ),
    )


def _timed(function: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    a.record()
    output = function()
    b.record()
    b.synchronize()
    return output, dict(
        cuda_ms=float(a.elapsed_time(b)), wall_ms=(time.perf_counter() - start) * 1000
    )


def benchmark(
    module: nn.Module,
    arguments: tuple[Tensor, ...],
    loss: Callable[[Any], Tensor],
    *,
    frames: int,
    microbatches: int,
    compile_mode: str,
    fullgraph: bool,
    warmup: int = 10,
    repetitions: int = 30,
    profiler_path: Path | None = None,
) -> dict[str, Any]:
    """One randomized-order block. Caller runs three blocks with reordered cases.

    Each measured backward has a new graph. Phases synchronize explicitly;
    total wall time includes these synchronization/launch costs. Input-pipeline
    time is excluded. Models start random; benchmark optimizer updates are not
    scientific training and no accuracy is returned.
    """
    if warmup < 10 or repetitions < 30 or frames < 1 or microbatches < 1:
        raise ValueError(
            "Require warmup>=10, repetitions>=30 and positive shape/update units"
        )
    if next(module.parameters()).device.type != "cuda":
        raise ValueError("Benchmark module must be on CUDA")
    compile_coordinate_model(module, mode=compile_mode, fullgraph=fullgraph)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=0.01)
    raw: dict[str, list[dict[str, Any]]] = {"inference": [], "training": []}
    cold = {}

    def inference() -> dict[str, Any]:
        module.eval()
        with (
            torch.inference_mode(),
            coordinate_compile_scope(module),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            _, result = _timed(lambda: module(*arguments))
        return result

    def update() -> dict[str, Any]:
        module.train()
        optimizer.zero_grad(set_to_none=True)
        for tensor in arguments:
            if tensor.requires_grad:
                tensor.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event, end_event = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        started = time.perf_counter()
        start_event.record()
        phases: dict[str, dict[str, float]] = {}

        def add(name: str, values: dict[str, float]) -> None:
            target = phases.setdefault(name, dict(cuda_ms=0.0, wall_ms=0.0))
            for k, value in values.items():
                target[k] += value

        with coordinate_compile_scope(module):
            for _ in range(microbatches):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, timing = _timed(lambda: module(*arguments))
                    add("forward", timing)
                    objective, timing = _timed(partial(scaled_loss, prediction))
                    add("loss", timing)
                _, timing = _timed(objective.backward)
                add("backward", timing)
            _, phases["gradient_clip"] = _timed(
                lambda: torch.nn.utils.clip_grad_norm_(
                    module.parameters(), 1.0, error_if_nonfinite=True
                )
            )
            _, phases["optimizer"] = _timed(optimizer.step)
        end_event.record()
        end_event.synchronize()
        return dict(
            phases=phases,
            total=dict(
                cuda_ms=float(start_event.elapsed_time(end_event)),
                wall_ms=(time.perf_counter() - started) * 1000,
            ),
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            parameter_bytes=tensor_bytes(list(module.parameters())),
            gradient_bytes=tensor_bytes([p.grad for p in module.parameters()]),
            optimizer_bytes=tensor_bytes(optimizer.state_dict()),
            backward_forward_ratio=phases["backward"]["cuda_ms"]
            / phases["forward"]["cuda_ms"],
        )

    def scaled_loss(output: Any) -> Tensor:
        return loss(output) / microbatches

    torch.cuda.reset_peak_memory_stats()
    cold["inference"] = inference()
    for _ in range(warmup):
        inference()
    for _ in range(repetitions):
        raw["inference"].append(inference())
    inference_peak = dict(
        allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved(),
    )
    cold["training"] = update()
    for _ in range(warmup):
        update()
    for _ in range(repetitions):
        raw["training"].append(update())
    if profiler_path is not None:
        profiler_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            update()
        profile.export_chrome_trace(str(profiler_path))
        (profiler_path.parent / (profiler_path.stem + "-ops.txt")).write_text(
            profile.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=150
            )
        )
    return dict(
        kind="untrained benchmark",
        raw=raw,
        cold_start=cold,
        warmup=warmup,
        repetitions=repetitions,
        frames_per_inference=frames,
        frames_per_optimizer_update=frames * microbatches,
        inference_peak=inference_peak,
        parameters=sum(p.numel() for p in module.parameters()),
        trainable_parameters=sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
        compilation=coordinate_compilation_report(module),
        summary=dict(
            inference_cuda_ms=quantiles(x["cuda_ms"] for x in raw["inference"]),
            inference_ms_per_frame=quantiles(
                x["cuda_ms"] / frames for x in raw["inference"]
            ),
            update_cuda_ms=quantiles(x["total"]["cuda_ms"] for x in raw["training"]),
            update_wall_ms=quantiles(x["total"]["wall_ms"] for x in raw["training"]),
        ),
    )


def conv_linear_macs(module: nn.Module, function: Callable[[], Any]) -> dict[str, Any]:
    """Same Conv2d/Conv3d/Linear contract as architecture.py; attention is explicit."""
    rows: dict[str, int] = {}
    handles = []

    def record(
        name: str, layer: nn.Module, _args: tuple[Tensor, ...], output: Tensor
    ) -> None:
        if isinstance(layer, (nn.Conv2d, nn.Conv3d)):
            value = (
                output.numel()
                * (layer.in_channels // layer.groups)
                * math.prod(layer.kernel_size)
            )
        elif isinstance(layer, nn.Linear):
            value = output.numel() * layer.in_features
        else:
            return
        rows[name] = rows.get(name, 0) + int(value)

    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(
                layer.register_forward_hook(
                    lambda m, args, out, name=name: record(name, m, args, out)
                )
            )
    try:
        function()
    finally:
        for handle in handles:
            handle.remove()
    return dict(
        conv_linear_macs=sum(rows.values()),
        layers=rows,
        excludes=[
            "normalization",
            "activation",
            "MDD",
            "memory transfers",
            "backward",
            "functional MHA projections and attention matmuls",
            "MSDA sampling (not zero)",
        ],
    )
