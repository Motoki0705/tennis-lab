"""Bounded retry policy for stochastic BLCS full-physics proposals."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    is_retryable_full_physics_rejection,
)

logger = logging.getLogger(__name__)

_SceneT = TypeVar("_SceneT")


def generate_with_bounded_physics_resampling(
    proposal: Callable[[], _SceneT | None],
    *,
    scene_id: str,
    maximum_attempts: int,
) -> _SceneT:
    """Return one accepted proposal within an explicit finite retry budget."""
    if (
        isinstance(maximum_attempts, bool)
        or not isinstance(maximum_attempts, int)
        or maximum_attempts <= 0
    ):
        raise ValueError("maximum_attempts must be a positive integer.")

    last_rejection: RuntimeError | None = None
    last_reason = "BLCS physical scene generation returned no scene."
    for attempt in range(1, maximum_attempts + 1):
        try:
            scene = proposal()
        except RuntimeError as error:
            if not is_retryable_full_physics_rejection(error):
                raise
            last_rejection = error
            last_reason = str(error)
            continue
        if scene is not None:
            if attempt > 1:
                logger.info(
                    "Accepted BLCS physics proposal for %s after bounded "
                    "resampling (attempt %s/%s); last_rejection=%s",
                    scene_id,
                    attempt,
                    maximum_attempts,
                    last_reason,
                )
            return scene
        last_rejection = None
        last_reason = "BLCS physical scene generation returned no scene."

    exhausted = RuntimeError(
        "BLCS physical scene generation exhausted "
        f"{maximum_attempts} bounded attempts for {scene_id!r}; "
        f"last_rejection={last_reason}"
    )
    if last_rejection is not None:
        raise exhausted from last_rejection
    raise exhausted


__all__ = ["generate_with_bounded_physics_resampling"]
