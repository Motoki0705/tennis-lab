"""Declared components, input assembly, clip storage and generic execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator

__all__ = ["TennisSceneOrchestrator"]


def __getattr__(name: str) -> type[TennisSceneOrchestrator]:
    # Task-owned models use artifact primitives; importing those must not build
    # the composition root, which imports the same task models.
    if name != "TennisSceneOrchestrator":
        raise AttributeError(name)
    from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator

    return cast(type[TennisSceneOrchestrator], TennisSceneOrchestrator)
