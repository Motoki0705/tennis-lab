"""Exact schema registry for versioned Court dataset artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.utils.schema.court import GROUND_COURT_KP_NAMES


class CourtDatasetSchemaVersion(StrEnum):
    """Explicit Court generation version selected at the config boundary."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


COURT_DATASET_SCHEMA_V1 = "canonical_court_dataset_v1"
COURT_DATASET_SCHEMA_V2 = "canonical_court_dataset_v2"
COURT_DATASET_SCHEMA_V3 = "canonical_court_dataset_v3"
COURT_PLAN_SCHEMA_V1 = "canonical_court_orbit_plan_v1"
COURT_PLAN_SCHEMA_V2 = "canonical_court_orbit_plan_v2"
COURT_PLAN_SCHEMA_V3 = "canonical_court_orbit_plan_v3"
COURT_SAMPLE_SCHEMA_V1 = "canonical_court_sample_v1"
COURT_SAMPLE_SCHEMA_V2 = "canonical_court_sample_v2"
COURT_SAMPLE_SCHEMA_V3 = "canonical_court_sample_v3"
COURT_SEMANTIC_MANIFEST_SCHEMA_V1 = "court_renderer_semantic_manifest_v1"
COURT_SEMANTIC_MANIFEST_SCHEMA_V2 = "court_renderer_semantic_manifest_v2"
COURT_SEMANTIC_MANIFEST_SCHEMA_V3 = "court_renderer_semantic_manifest_v3"
COURT_PERFORMANCE_SCHEMA_V1 = "court_dataset_performance_v2"
COURT_PERFORMANCE_SCHEMA_V2 = "court_dataset_performance_v3"
COURT_PERFORMANCE_SCHEMA_V3 = "court_dataset_performance_v4"
COURT_SHARD_SCHEMA_V1 = "court_render_shard_attempt_v1"
COURT_SHARD_SCHEMA_V2 = "court_render_shard_attempt_v2"
COURT_SHARD_SCHEMA_V3 = "court_render_shard_attempt_v3"
COURT_ARC_STEP_DIAGNOSTICS_SCHEMA_V1 = "court_arc_step_diagnostics_v1"
COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V1 = "court_acceptance_diagnostics_v1"
COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V2 = "court_acceptance_diagnostics_v2"
COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V3 = "court_acceptance_diagnostics_v3"
COURT_SPLIT_DIAGNOSTICS_SCHEMA_V1 = "court_split_diagnostics_v1"
COURT_SPLIT_DIAGNOSTICS_SCHEMA_V2 = "court_split_diagnostics_v2"
COURT_SPLIT_DIAGNOSTICS_SCHEMA_V3 = "court_split_diagnostics_v3"
COURT_PARAMETER_TABLE_SCHEMA_V1 = "court_parameter_table_v1"
COURT_PARAMETER_TABLE_SCHEMA_V2 = "court_parameter_table_v2"
COURT_PARAMETER_TABLE_SCHEMA_V3 = "court_parameter_table_v3"
COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V1 = (
    "court_semantic_visibility_diagnostics_v1"
)
COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V2 = (
    "court_semantic_visibility_diagnostics_v2"
)
COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V3 = (
    "court_semantic_visibility_diagnostics_v3"
)

COURT_SEMANTIC_CLASS_NAMES_V1: tuple[str, ...] = (
    "doubles_left",
    "doubles_right",
    "singles_left",
    "singles_right",
    "service_left",
    "service_right",
    "service_t",
)
COURT_PHYSICAL_INDICES_BY_CLASS_V1: tuple[tuple[int, ...], ...] = (
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (8, 10),
    (9, 11),
    (12, 13),
)
COURT_SEMANTIC_CLASS_NAMES_V2: tuple[str, ...] = GROUND_COURT_KP_NAMES
COURT_SEMANTIC_CLASS_NAMES_V3: tuple[str, ...] = GROUND_COURT_KP_NAMES


@dataclass(frozen=True, slots=True)
class CourtSchemaDefinition:
    """All exact public schema identifiers and semantic cardinalities."""

    version: CourtDatasetSchemaVersion
    dataset_schema: str
    plan_schema: str
    sample_schema: str
    semantic_manifest_schema: str
    performance_schema: str
    shard_schema: str
    arc_step_diagnostics_schema: str
    acceptance_diagnostics_schema: str
    split_diagnostics_schema: str
    parameter_table_schema: str
    semantic_visibility_diagnostics_schema: str
    semantic_class_names: tuple[str, ...]
    points_per_class: int

    @property
    def semantic_class_count(self) -> int:
        """Return the number of ordered semantic channels."""
        return len(self.semantic_class_names)


COURT_SCHEMA_V1 = CourtSchemaDefinition(
    version=CourtDatasetSchemaVersion.V1,
    dataset_schema=COURT_DATASET_SCHEMA_V1,
    plan_schema=COURT_PLAN_SCHEMA_V1,
    sample_schema=COURT_SAMPLE_SCHEMA_V1,
    semantic_manifest_schema=COURT_SEMANTIC_MANIFEST_SCHEMA_V1,
    performance_schema=COURT_PERFORMANCE_SCHEMA_V1,
    shard_schema=COURT_SHARD_SCHEMA_V1,
    arc_step_diagnostics_schema=COURT_ARC_STEP_DIAGNOSTICS_SCHEMA_V1,
    acceptance_diagnostics_schema=COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V1,
    split_diagnostics_schema=COURT_SPLIT_DIAGNOSTICS_SCHEMA_V1,
    parameter_table_schema=COURT_PARAMETER_TABLE_SCHEMA_V1,
    semantic_visibility_diagnostics_schema=(
        COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V1
    ),
    semantic_class_names=COURT_SEMANTIC_CLASS_NAMES_V1,
    points_per_class=2,
)
COURT_SCHEMA_V2 = CourtSchemaDefinition(
    version=CourtDatasetSchemaVersion.V2,
    dataset_schema=COURT_DATASET_SCHEMA_V2,
    plan_schema=COURT_PLAN_SCHEMA_V2,
    sample_schema=COURT_SAMPLE_SCHEMA_V2,
    semantic_manifest_schema=COURT_SEMANTIC_MANIFEST_SCHEMA_V2,
    performance_schema=COURT_PERFORMANCE_SCHEMA_V2,
    shard_schema=COURT_SHARD_SCHEMA_V2,
    arc_step_diagnostics_schema=COURT_ARC_STEP_DIAGNOSTICS_SCHEMA_V1,
    acceptance_diagnostics_schema=COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V2,
    split_diagnostics_schema=COURT_SPLIT_DIAGNOSTICS_SCHEMA_V2,
    parameter_table_schema=COURT_PARAMETER_TABLE_SCHEMA_V2,
    semantic_visibility_diagnostics_schema=(
        COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V2
    ),
    semantic_class_names=COURT_SEMANTIC_CLASS_NAMES_V2,
    points_per_class=1,
)
COURT_SCHEMA_V3 = CourtSchemaDefinition(
    version=CourtDatasetSchemaVersion.V3,
    dataset_schema=COURT_DATASET_SCHEMA_V3,
    plan_schema=COURT_PLAN_SCHEMA_V3,
    sample_schema=COURT_SAMPLE_SCHEMA_V3,
    semantic_manifest_schema=COURT_SEMANTIC_MANIFEST_SCHEMA_V3,
    performance_schema=COURT_PERFORMANCE_SCHEMA_V3,
    shard_schema=COURT_SHARD_SCHEMA_V3,
    arc_step_diagnostics_schema=COURT_ARC_STEP_DIAGNOSTICS_SCHEMA_V1,
    acceptance_diagnostics_schema=COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V3,
    split_diagnostics_schema=COURT_SPLIT_DIAGNOSTICS_SCHEMA_V3,
    parameter_table_schema=COURT_PARAMETER_TABLE_SCHEMA_V3,
    semantic_visibility_diagnostics_schema=(
        COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V3
    ),
    semantic_class_names=COURT_SEMANTIC_CLASS_NAMES_V3,
    points_per_class=1,
)

COURT_SCHEMA_BY_VERSION = {
    CourtDatasetSchemaVersion.V1: COURT_SCHEMA_V1,
    CourtDatasetSchemaVersion.V2: COURT_SCHEMA_V2,
    CourtDatasetSchemaVersion.V3: COURT_SCHEMA_V3,
}
COURT_SCHEMA_BY_DATASET_SCHEMA = {
    definition.dataset_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}
COURT_SCHEMA_BY_PLAN_SCHEMA = {
    definition.plan_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}
COURT_SCHEMA_BY_SAMPLE_SCHEMA = {
    definition.sample_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}
COURT_SCHEMA_BY_SEMANTIC_MANIFEST_SCHEMA = {
    definition.semantic_manifest_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}
COURT_SCHEMA_BY_PERFORMANCE_SCHEMA = {
    definition.performance_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}
COURT_SCHEMA_BY_SHARD_SCHEMA = {
    definition.shard_schema: definition
    for definition in COURT_SCHEMA_BY_VERSION.values()
}


def court_schema_for_version(
    version: CourtDatasetSchemaVersion,
) -> CourtSchemaDefinition:
    """Resolve an explicitly typed version without string or shape fallback."""
    if not isinstance(version, CourtDatasetSchemaVersion):
        raise TypeError("version must be a CourtDatasetSchemaVersion.")
    return COURT_SCHEMA_BY_VERSION[version]


def court_schema_from_dataset_schema(schema: object) -> CourtSchemaDefinition:
    """Dispatch one dataset reader from its exact top-level schema string."""
    if not isinstance(schema, str):
        raise TypeError("Court dataset schema must be a string.")
    try:
        return COURT_SCHEMA_BY_DATASET_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(f"Unknown Court dataset schema: {schema!r}.") from error


def court_schema_from_plan_schema(schema: object) -> CourtSchemaDefinition:
    """Dispatch one plan reader from its exact top-level schema string."""
    if not isinstance(schema, str):
        raise TypeError("Court plan schema must be a string.")
    try:
        return COURT_SCHEMA_BY_PLAN_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(f"Unknown Court plan schema: {schema!r}.") from error


def court_schema_from_sample_schema(schema: object) -> CourtSchemaDefinition:
    """Dispatch one sample reader from its exact top-level schema string."""
    if not isinstance(schema, str):
        raise TypeError("Court sample schema must be a string.")
    try:
        return COURT_SCHEMA_BY_SAMPLE_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(f"Unknown Court sample schema: {schema!r}.") from error


def court_schema_from_semantic_manifest_schema(
    schema: object,
) -> CourtSchemaDefinition:
    """Dispatch a semantic manifest from its exact top-level schema."""
    if not isinstance(schema, str):
        raise TypeError("Court semantic manifest schema must be a string.")
    try:
        return COURT_SCHEMA_BY_SEMANTIC_MANIFEST_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(
            f"Unknown Court semantic manifest schema: {schema!r}."
        ) from error


def court_schema_from_performance_schema(schema: object) -> CourtSchemaDefinition:
    """Dispatch performance evidence from its exact top-level schema."""
    if not isinstance(schema, str):
        raise TypeError("Court performance schema must be a string.")
    try:
        return COURT_SCHEMA_BY_PERFORMANCE_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(f"Unknown Court performance schema: {schema!r}.") from error


def court_schema_from_shard_schema(schema: object) -> CourtSchemaDefinition:
    """Dispatch an attempt-local shard from its exact top-level schema."""
    if not isinstance(schema, str):
        raise TypeError("Court shard schema must be a string.")
    try:
        return COURT_SCHEMA_BY_SHARD_SCHEMA[schema]
    except KeyError as error:
        raise ValueError(f"Unknown Court shard schema: {schema!r}.") from error


__all__ = [
    "COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V1",
    "COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V2",
    "COURT_ACCEPTANCE_DIAGNOSTICS_SCHEMA_V3",
    "COURT_ARC_STEP_DIAGNOSTICS_SCHEMA_V1",
    "COURT_DATASET_SCHEMA_V1",
    "COURT_DATASET_SCHEMA_V2",
    "COURT_DATASET_SCHEMA_V3",
    "COURT_PHYSICAL_INDICES_BY_CLASS_V1",
    "COURT_PARAMETER_TABLE_SCHEMA_V1",
    "COURT_PARAMETER_TABLE_SCHEMA_V2",
    "COURT_PARAMETER_TABLE_SCHEMA_V3",
    "COURT_PERFORMANCE_SCHEMA_V1",
    "COURT_PERFORMANCE_SCHEMA_V2",
    "COURT_PERFORMANCE_SCHEMA_V3",
    "COURT_PLAN_SCHEMA_V1",
    "COURT_PLAN_SCHEMA_V2",
    "COURT_PLAN_SCHEMA_V3",
    "COURT_SAMPLE_SCHEMA_V1",
    "COURT_SAMPLE_SCHEMA_V2",
    "COURT_SAMPLE_SCHEMA_V3",
    "COURT_SCHEMA_BY_DATASET_SCHEMA",
    "COURT_SCHEMA_BY_PLAN_SCHEMA",
    "COURT_SCHEMA_BY_PERFORMANCE_SCHEMA",
    "COURT_SCHEMA_BY_SAMPLE_SCHEMA",
    "COURT_SCHEMA_BY_SEMANTIC_MANIFEST_SCHEMA",
    "COURT_SCHEMA_BY_SHARD_SCHEMA",
    "COURT_SCHEMA_BY_VERSION",
    "COURT_SCHEMA_V1",
    "COURT_SCHEMA_V2",
    "COURT_SCHEMA_V3",
    "COURT_SEMANTIC_CLASS_NAMES_V1",
    "COURT_SEMANTIC_CLASS_NAMES_V2",
    "COURT_SEMANTIC_CLASS_NAMES_V3",
    "COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V1",
    "COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V2",
    "COURT_SEMANTIC_VISIBILITY_DIAGNOSTICS_SCHEMA_V3",
    "COURT_SEMANTIC_MANIFEST_SCHEMA_V1",
    "COURT_SEMANTIC_MANIFEST_SCHEMA_V2",
    "COURT_SEMANTIC_MANIFEST_SCHEMA_V3",
    "COURT_SHARD_SCHEMA_V1",
    "COURT_SHARD_SCHEMA_V2",
    "COURT_SHARD_SCHEMA_V3",
    "COURT_SPLIT_DIAGNOSTICS_SCHEMA_V1",
    "COURT_SPLIT_DIAGNOSTICS_SCHEMA_V2",
    "COURT_SPLIT_DIAGNOSTICS_SCHEMA_V3",
    "CourtDatasetSchemaVersion",
    "CourtSchemaDefinition",
    "court_schema_for_version",
    "court_schema_from_dataset_schema",
    "court_schema_from_plan_schema",
    "court_schema_from_performance_schema",
    "court_schema_from_sample_schema",
    "court_schema_from_semantic_manifest_schema",
    "court_schema_from_shard_schema",
]
