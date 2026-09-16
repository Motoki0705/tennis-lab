"""Composition root for Court input implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from src.tasks.court_detection.configuration import (
    CourtSourceConfig,
    SyntheticCourtSourceConfig,
    TennisCourtDetectorSourceConfig,
)
from src.tasks.court_detection.data.contracts import CourtSourceSplit
from src.tasks.court_detection.data.inputs.contract import CourtInput
from src.tasks.court_detection.data.inputs.synthetic_court import SyntheticCourtInput
from src.tasks.court_detection.data.inputs.tennis_court_detector import (
    TennisCourtDetectorInput,
)
from src.tasks.court_detection.data.target_generation.store import (
    CourtDerivedTargetStore,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    line_target_definition,
)


def _validate_external_store(
    config: CourtSourceConfig,
    store: CourtDerivedTargetStore,
) -> None:
    source_root = (
        config.root
        if isinstance(config, TennisCourtDetectorSourceConfig)
        else config.workspace_root
    )
    if store.root.resolve(strict=False).is_relative_to(
        source_root.resolve(strict=False)
    ):
        raise ValueError(
            "Court derived_target_root must be outside the selected source root."
        )


def _build_tennis(
    config: CourtSourceConfig,
    store: CourtDerivedTargetStore,
    line_target_schema: str,
    requested_splits: Sequence[CourtSourceSplit] | None,
) -> CourtInput:
    return TennisCourtDetectorInput(
        cast(TennisCourtDetectorSourceConfig, config),
        target_store=store,
        line_target_schema=line_target_schema,
        requested_splits=requested_splits,
    )


def _build_synthetic(
    config: CourtSourceConfig,
    store: CourtDerivedTargetStore,
    line_target_schema: str,
    requested_splits: Sequence[CourtSourceSplit] | None,
) -> CourtInput:
    if requested_splits is not None:
        raise ValueError(
            "Synthetic Court input has no partial split read; requested_splits is "
            "only supported by the tennis_court_detector input."
        )
    return SyntheticCourtInput(
        cast(SyntheticCourtSourceConfig, config),
        target_store=store,
        line_target_schema=line_target_schema,
    )


_BUILDERS: dict[
    str,
    Callable[
        [
            CourtSourceConfig,
            CourtDerivedTargetStore,
            str,
            Sequence[CourtSourceSplit] | None,
        ],
        CourtInput,
    ],
] = {
    "tennis_court_detector": _build_tennis,
    "synthetic_court": _build_synthetic,
}


def build_court_input(
    config: CourtSourceConfig,
    *,
    target_store: CourtDerivedTargetStore,
    line_target_schema: str = LINE_TARGET_SCHEMA,
    requested_splits: Sequence[CourtSourceSplit] | None = None,
) -> CourtInput:
    """Resolve the explicit source discriminator exactly once.

    ``requested_splits`` is forwarded to inputs that can preflight a subset of
    their configured splits; omitting it keeps the full read every existing
    caller relied on.
    """
    _validate_external_store(config, target_store)
    line_target_definition(line_target_schema)
    try:
        builder = _BUILDERS[config.kind]
    except KeyError as error:  # defensive: typed configuration already validates
        raise ValueError(f"Unsupported Court input kind: {config.kind!r}.") from error
    return builder(config, target_store, line_target_schema, requested_splits)


__all__ = ["build_court_input"]
