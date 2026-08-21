"""Exact dispatch tests for every versioned Court schema boundary."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SCHEMA_V1,
    COURT_SCHEMA_V2,
    CourtDatasetSchemaVersion,
    court_schema_for_version,
    court_schema_from_dataset_schema,
    court_schema_from_performance_schema,
    court_schema_from_plan_schema,
    court_schema_from_sample_schema,
    court_schema_from_semantic_manifest_schema,
    court_schema_from_shard_schema,
)
from src.utils.schema.court import COURT_KP_NAMES


def test_v1_and_v2_registry_exposes_all_exact_boundary_schemas() -> None:
    expected = {
        CourtDatasetSchemaVersion.V1: (
            COURT_SCHEMA_V1,
            (
                "canonical_court_dataset_v1",
                "canonical_court_orbit_plan_v1",
                "canonical_court_sample_v1",
                "court_renderer_semantic_manifest_v1",
                "court_dataset_performance_v2",
                "court_render_shard_attempt_v1",
            ),
            7,
            2,
        ),
        CourtDatasetSchemaVersion.V2: (
            COURT_SCHEMA_V2,
            (
                "canonical_court_dataset_v2",
                "canonical_court_orbit_plan_v2",
                "canonical_court_sample_v2",
                "court_renderer_semantic_manifest_v2",
                "court_dataset_performance_v3",
                "court_render_shard_attempt_v2",
            ),
            14,
            1,
        ),
    }
    readers = (
        court_schema_from_dataset_schema,
        court_schema_from_plan_schema,
        court_schema_from_sample_schema,
        court_schema_from_semantic_manifest_schema,
        court_schema_from_performance_schema,
        court_schema_from_shard_schema,
    )

    for version, (
        definition,
        schemas,
        class_count,
        points_per_class,
    ) in expected.items():
        assert court_schema_for_version(version) is definition
        assert definition.semantic_class_count == class_count
        assert definition.points_per_class == points_per_class
        assert tuple(
            reader(schema) for reader, schema in zip(readers, schemas, strict=True)
        ) == (definition,) * len(readers)

    assert COURT_SCHEMA_V2.semantic_class_names == COURT_KP_NAMES[:14]


@pytest.mark.parametrize(
    "reader",
    [
        court_schema_from_dataset_schema,
        court_schema_from_plan_schema,
        court_schema_from_sample_schema,
        court_schema_from_semantic_manifest_schema,
        court_schema_from_performance_schema,
        court_schema_from_shard_schema,
    ],
)
def test_schema_dispatch_never_infers_unknown_or_non_string_versions(reader) -> None:
    with pytest.raises(ValueError, match="Unknown Court"):
        reader("canonical_court_payload_v3")
    with pytest.raises(TypeError, match="must be a string"):
        reader(None)


def test_version_lookup_requires_the_typed_selector() -> None:
    with pytest.raises(TypeError, match="CourtDatasetSchemaVersion"):
        court_schema_for_version("v2")  # type: ignore[arg-type]
