"""Versioned semantic contracts for derived Court targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

SEGMENTATION_TARGET_SCHEMA_HARD = "court_cell_segmentation_single_court_v2"
SEGMENTATION_TARGET_SCHEMA = "court_cell_segmentation_coverage8_single_court_v3"

SEMANTIC_LINE_TARGET_SCHEMA_HARD = (
    "court_line_semantic_camera_view_75mm_150mm_single_court_v1"
)
SEMANTIC_LINE_TARGET_SCHEMA = (
    "court_line_semantic_camera_view_75mm_150mm_coverage8_single_court_v2"
)
DENSE_COVERAGE_FACTOR = 8
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


LINE_TARGET_SCHEMA_V3 = "court_line_binary_75mm_150mm_single_court_v3"
LINE_TARGET_SCHEMA_HARD = LINE_TARGET_SCHEMA_V3
LINE_TARGET_SCHEMA = "court_line_coverage8_75mm_150mm_single_court_v4"

DENSE_COVERAGE_SCHEMA_BY_KIND = {
    "seg": SEGMENTATION_TARGET_SCHEMA,
    "line": LINE_TARGET_SCHEMA,
    "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA,
}
HARD_DENSE_SCHEMA_BY_KIND = {
    "seg": SEGMENTATION_TARGET_SCHEMA_HARD,
    "line": LINE_TARGET_SCHEMA_HARD,
    "semantic_line": SEMANTIC_LINE_TARGET_SCHEMA_HARD,
}
DENSE_COVERAGE_SCHEMAS = frozenset(DENSE_COVERAGE_SCHEMA_BY_KIND.values())


SEMANTIC_LINE_TARGET_DEFINITION = CourtLineTargetDefinition(
    schema=SEMANTIC_LINE_TARGET_SCHEMA,
    line_width_metres=0.075,
    baseline_width_metres=0.15,
)

LINE_TARGET_DEFINITIONS: Mapping[str, CourtLineTargetDefinition] = MappingProxyType(
    {
        LINE_TARGET_SCHEMA_V3: CourtLineTargetDefinition(
            schema=LINE_TARGET_SCHEMA_V3,
            line_width_metres=0.075,
            baseline_width_metres=0.15,
        ),
        LINE_TARGET_SCHEMA: CourtLineTargetDefinition(
            schema=LINE_TARGET_SCHEMA,
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
        raise ValueError(
            f"Unsupported Court line target schema: {schema!r}."
        ) from error


__all__ = [
    "DENSE_COVERAGE_FACTOR",
    "DENSE_COVERAGE_SCHEMA_BY_KIND",
    "DENSE_COVERAGE_SCHEMAS",
    "HARD_DENSE_SCHEMA_BY_KIND",
    "LINE_TARGET_SCHEMA_HARD",
    "SEGMENTATION_TARGET_SCHEMA_HARD",
    "SEMANTIC_LINE_TARGET_SCHEMA_HARD",
    "LINE_TARGET_DEFINITIONS",
    "LINE_TARGET_SCHEMA",
    "LINE_TARGET_SCHEMA_V3",
    "SEMANTIC_LINE_CHANNEL_NAMES",
    "SEMANTIC_LINE_CLASS_BY_NAME",
    "SEMANTIC_LINE_TARGET_DEFINITION",
    "SEMANTIC_LINE_TARGET_SCHEMA",
    "SEGMENTATION_TARGET_SCHEMA",
    "CourtLineTargetDefinition",
    "line_target_definition",
]
