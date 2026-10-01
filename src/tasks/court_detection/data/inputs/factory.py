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


def _build_tennis(
    config: CourtSourceConfig,
) -> CourtInput:
    return TennisCourtDetectorInput(
        cast(TennisCourtDetectorSourceConfig, config),
    )


def _build_synthetic(
    config: CourtSourceConfig,
) -> CourtInput:
    return SyntheticCourtInput(
        cast(SyntheticCourtSourceConfig, config),
    )


_BUILDERS: dict[
    str,
    Callable[[CourtSourceConfig], CourtInput],
] = {
    "tennis_court_detector": _build_tennis,
    "synthetic_court": _build_synthetic,
}


def build_court_input(
    config: CourtSourceConfig,
) -> CourtInput:
    """Resolve the explicit source discriminator exactly once."""
    try:
        builder = _BUILDERS[config.kind]
    except KeyError as error:  # defensive: typed configuration already validates
        raise ValueError(f"Unsupported Court input kind: {config.kind!r}.") from error
    return builder(config)


__all__ = ["build_court_input"]
