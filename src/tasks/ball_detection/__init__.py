"""Ball detection public factory, loaded only when a model is requested."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair

__all__ = ["build_ball_detection_pair"]


def __getattr__(name: str) -> Any:
    if name != "build_ball_detection_pair":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair

    globals()[name] = build_ball_detection_pair
    return build_ball_detection_pair
