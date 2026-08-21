"""Shared, explicit ``torch.compile`` dispatch for task training models."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from src.tasks.base.configuration import CompileConfig


class CompilationTargetError(RuntimeError):
    """Raised when a training module exposes an invalid compile target contract."""


def compile_modules(
    targets: Mapping[str, nn.Module],
    config: CompileConfig,
) -> tuple[str, ...]:
    """Compile each distinct named module in place and return compiled names.

    ``nn.Module.compile`` preserves module identity and state-dict keys.  Targets
    are explicit because recursively compiling every child of a LightningModule
    would also capture losses/metrics and could compile model children twice.
    """
    if not config.enabled:
        return ()
    if not targets:
        raise CompilationTargetError(
            "training.compile.enabled=true requires at least one compile target."
        )

    compiled_names: list[str] = []
    seen_modules: set[int] = set()
    for name, module in targets.items():
        if not isinstance(name, str) or not name or name != name.strip():
            raise CompilationTargetError(
                "Compile target names must be non-empty trimmed strings."
            )
        if not isinstance(module, nn.Module):
            raise CompilationTargetError(
                f"Compile target {name!r} must be an nn.Module, got "
                f"{type(module).__name__}."
            )
        identity = id(module)
        if identity in seen_modules:
            continue
        seen_modules.add(identity)
        module.compile(
            backend=config.backend,
            mode=config.mode,
            fullgraph=config.fullgraph,
            dynamic=config.dynamic,
        )
        compiled_names.append(name)

    rendered = ", ".join(compiled_names)
    print(
        "[torch.compile] enabled "
        f"targets=[{rendered}] backend={config.backend} mode={config.mode} "
        f"fullgraph={config.fullgraph} dynamic={config.dynamic}"
    )
    return tuple(compiled_names)


__all__ = ["CompilationTargetError", "compile_modules"]
