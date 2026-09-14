"""Visualization utilities for PLCS.

The renderer exports pull in the training stack (``pytorch_lightning``), so
they are resolved on first attribute access. Lightweight callers such as the
read-only ``review`` server can import a submodule without paying for it.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tasks.plcs.visualization.contracts import PoseRenderScene
    from src.tasks.plcs.visualization.orchestrator import (
        RuntimeConfig,
        build_runtime_config,
        run_visualization,
    )
    from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer

__all__ = [
    "PLCSSceneRenderer",
    "PoseRenderScene",
    "RuntimeConfig",
    "build_runtime_config",
    "run_visualization",
]

_ORCHESTRATOR = "src.tasks.plcs.visualization.orchestrator"
_EXPORTS: dict[str, str] = {
    "PoseRenderScene": "src.tasks.plcs.visualization.contracts",
    "RuntimeConfig": _ORCHESTRATOR,
    "build_runtime_config": _ORCHESTRATOR,
    "run_visualization": _ORCHESTRATOR,
    "PLCSSceneRenderer": "src.tasks.plcs.visualization.rendering",
}


def __getattr__(name: str) -> object:
    """Resolve a public export from its owning module on first use."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
