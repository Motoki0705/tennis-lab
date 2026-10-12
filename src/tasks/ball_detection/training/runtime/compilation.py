"""Strict model compilation with the coordinate trainer's AMP/backward policy."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn

COMPILE_MODES = ("off", "default", "reduce-overhead", "max-autotune")


def compile_coordinate_model(model: nn.Module, *, mode: str, recompile_limit: int = 8) -> None:
    if mode not in COMPILE_MODES or recompile_limit < 1:
        raise ValueError("Invalid coordinate compilation mode or recompile limit")
    if getattr(model, "_compiled_call_impl", None) is not None:
        raise ValueError("Model was already compiled; configure its runtime exactly once")
    # Execution metadata is neither a parameter nor a registered child module.
    model.__dict__["_coordinate_compile_mode"] = mode
    model.__dict__["_coordinate_recompile_limit"] = recompile_limit
    if mode == "off":
        return
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("Coordinate model compilation is supported on CUDA; no eager fallback")
    # In-place compilation preserves model type and unprefixed state_dict names.
    with coordinate_compile_scope(model):
        model.compile(backend="inductor", mode=mode, fullgraph=True, dynamic=False,
                      recompile_limit=recompile_limit)


@contextmanager
def coordinate_compile_scope(model: nn.Module) -> Iterator[None]:
    if getattr(model, "_coordinate_compile_mode", "off") == "off":
        yield
        return
    import torch._dynamo.config as dynamo_config
    import torch._functorch.config as aot_config

    # Compilation is lazy: keep the policy active during actual forward/backward,
    # not only when Module.compile() installs its callable. Backward runs outside
    # autocast in this trainer. Defaults assuming forward's autocast are unsafe.
    with dynamo_config.patch(suppress_errors=False), aot_config.patch(backward_pass_autocast="off"):
        yield


def coordinate_compilation_report(model: nn.Module) -> dict[str, Any]:
    mode = getattr(model, "_coordinate_compile_mode", "off")
    report: dict[str, Any] = dict(mode=mode, backend="inductor" if mode != "off" else None,
                                 fullgraph=mode != "off", dynamic=False,
                                 recompile_limit=getattr(model, "_coordinate_recompile_limit", 8),
                                 backward_pass_autocast="off")
    if mode != "off":
        from torch._dynamo.utils import counters

        report.update(unique_graphs=int(counters["stats"]["unique_graphs"]),
                      graph_breaks=dict(counters["graph_break"]))
    return report
