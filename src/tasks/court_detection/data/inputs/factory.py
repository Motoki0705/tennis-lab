"""Composition root for Court input implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from src.tasks.court_detection.configuration import (
    CourtSourceConfig,
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)


def _build_tennis(
    config: CourtSourceConfig, store: CourtDerivedTargetStore
) -> CourtInput:
    return TennisCourtDetectorInput(
        cast(TennisCourtDetectorSourceConfig, config), target_store=store
    )


def _build_synthetic(
    config: CourtSourceConfig, store: CourtDerivedTargetStore
) -> CourtInput:
    return SyntheticCourtInput(
        cast(SyntheticCourtSourceConfig, config), target_store=store
    )


_BUILDERS: dict[str, Callable[[CourtSourceConfig, CourtDerivedTargetStore], CourtInput]] = {
    "tennis_court_detector": _build_tennis,
    "synthetic_court": _build_synthetic,
}


def build_court_input(
    config: CourtSourceConfig,
    *,
    target_store: CourtDerivedTargetStore,
) -> CourtInput:
    """Resolve the explicit source discriminator exactly once."""
    try:
        builder = _BUILDERS[config.kind]
    except KeyError as error:  # defensive: typed configuration already validates
        raise ValueError(f"Unsupported Court input kind: {config.kind!r}.") from error
    return builder(config, target_store)


__all__ = ["build_court_input"]
