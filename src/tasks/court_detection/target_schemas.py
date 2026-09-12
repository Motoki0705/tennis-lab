"""Versioned semantic contracts for derived Court targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

SEGMENTATION_TARGET_SCHEMA_V1 = "court_cell_segmentation_v1"
SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_single_court_v2"

SEMANTIC_LINE_TARGET_SCHEMA = (
    "court_line_semantic_camera_view_75mm_150mm_single_court_v1"
)
SEMANTIC_LINE_CHANNEL_NAMES = (
    "background",
    "far_baseline",
    "near_baseline",
    "left_doubles_sideline",
    "right_doubles_sideline",
    "left_singles_sideline",
    "right_singles_sideline",
    "far_service_line",
    "near_service_line",
    "center_service_line",
    "far_center_mark",
    "near_center_mark",
)
SEMANTIC_LINE_CLASS_BY_NAME: Mapping[str, int] = MappingProxyType(
    {name: index for index, name in enumerate(SEMANTIC_LINE_CHANNEL_NAMES)}
)


@dataclass(frozen=True, slots=True)
class CourtLineTargetDefinition:
    """Physical widths owned by one immutable binary-line target schema."""

    schema: str
    line_width_metres: float
    baseline_width_metres: float


LINE_TARGET_SCHEMA_V1 = "court_line_binary_v1"
LINE_TARGET_SCHEMA_V2 = "court_line_binary_75mm_150mm_v2"
LINE_TARGET_SCHEMA_V3 = "court_line_binary_75mm_150mm_single_court_v3"
LINE_TARGET_SCHEMA = LINE_TARGET_SCHEMA_V3

SEMANTIC_LINE_TARGET_DEFINITION = CourtLineTargetDefinition(
    schema=SEMANTIC_LINE_TARGET_SCHEMA,
    line_width_metres=0.075,
    baseline_width_metres=0.15,
)

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
        LINE_TARGET_SCHEMA_V3: CourtLineTargetDefinition(
            schema=LINE_TARGET_SCHEMA_V3,
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
    "LINE_TARGET_SCHEMA_V3",
    "SEMANTIC_LINE_CHANNEL_NAMES",
    "SEMANTIC_LINE_CLASS_BY_NAME",
    "SEMANTIC_LINE_TARGET_DEFINITION",
    "SEMANTIC_LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA_V1",
    "CourtLineTargetDefinition",
    "line_target_definition",
]
