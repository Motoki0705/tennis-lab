"""Versioned semantic contracts for derived Court targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_v1"


@dataclass(frozen=True, slots=True)
class CourtLineTargetDefinition:
    """Physical widths owned by one immutable binary-line target schema."""

    schema: str
    line_width_metres: float
    baseline_width_metres: float


LINE_TARGET_SCHEMA_V1 = "court_line_binary_v1"
LINE_TARGET_SCHEMA_V2 = "court_line_binary_75mm_150mm_v2"
LINE_TARGET_SCHEMA = LINE_TARGET_SCHEMA_V2

LINE_TARGET_DEFINITIONS: Mapping[str, CourtLineTargetDefinition] = MappingProxyType(
    {
        LINE_TARGET_SCHEMA_V1: CourtLineTargetDefinition(
            schema=LINE_TARGET_SCHEMA_V1,
            line_width_metres=0.05,
            baseline_width_metres=0.10,
        ),
        LINE_TARGET_SCHEMA_V2: CourtLineTargetDefinition(
            schema=LINE_TARGET_SCHEMA_V2,
            line_width_metres=0.075,
            baseline_width_metres=0.15,
        ),
    }
)


def line_target_definition(schema: str) -> CourtLineTargetDefinition:
    """Resolve a known line schema without silently changing its physical width."""
    try:
        return LINE_TARGET_DEFINITIONS[schema]
    except KeyError as error:
        raise ValueError(f"Unsupported Court line target schema: {schema!r}.") from error


__all__ = [
    "LINE_TARGET_DEFINITIONS",
    "LINE_TARGET_SCHEMA",
    "LINE_TARGET_SCHEMA_V1",
    "LINE_TARGET_SCHEMA_V2",
    "SEGMENTATION_TARGET_SCHEMA",
    "CourtLineTargetDefinition",
    "line_target_definition",
]
