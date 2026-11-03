"""Composition root for Court input implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from src.tasks.court_detection.configuration import (
    CourtSourceConfig,
    SyntheticCourtDatasetSubsetSourceConfig,
    SyntheticCourtDatasetVersionSourceConfig,
    SyntheticCourtSourceConfig,
    SyntheticCourtStoresSourceConfig,
    TennisCourtDetectorSourceConfig,
    TennisCourtEvaluationSourceConfig,
)
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.synthetic_court_dataset_version import (
    SyntheticCourtDatasetVersionInput,
)
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
        cast(SyntheticCourtSourceConfig | SyntheticCourtStoresSourceConfig, config),
    )


def _build_evaluation(config: CourtSourceConfig) -> CourtInput:
    from src.tasks.court_detection.scaling.real_split import TennisCourtEvaluationInput

    return TennisCourtEvaluationInput(cast(TennisCourtEvaluationSourceConfig, config))


def _build_dataset_subset(config: CourtSourceConfig) -> CourtInput:
    from src.tasks.court_detection.data.inputs.synthetic_court_dataset_subset import (
        SyntheticCourtDatasetSubsetInput,
    )

    return SyntheticCourtDatasetSubsetInput(
        cast(SyntheticCourtDatasetSubsetSourceConfig, config)
    )


def _build_dataset_version(config: CourtSourceConfig) -> CourtInput:
    return SyntheticCourtDatasetVersionInput(
        cast(SyntheticCourtDatasetVersionSourceConfig, config)
    )


_BUILDERS: dict[
    str,
    Callable[[CourtSourceConfig], CourtInput],
] = {
    "tennis_court_detector": _build_tennis,
    "synthetic_court": _build_synthetic,
    "synthetic_court_stores": _build_synthetic,
    "synthetic_court_dataset_version": _build_dataset_version,
    "synthetic_court_dataset_subset": _build_dataset_subset,
    "tennis_court_evaluation": _build_evaluation,
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
