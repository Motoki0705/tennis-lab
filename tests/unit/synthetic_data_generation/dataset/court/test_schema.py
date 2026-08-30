"""Exact dispatch tests for every versioned Court schema boundary."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.court import (
    COURT_SCHEMA_V4 as EXPORTED_COURT_SCHEMA_V4,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SCHEMA_V1,
    COURT_SCHEMA_V2,
    COURT_SCHEMA_V3,
    COURT_SCHEMA_V4,
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


def test_version_registry_exposes_all_exact_boundary_schemas() -> None:
    assert EXPORTED_COURT_SCHEMA_V4 is COURT_SCHEMA_V4
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
        CourtDatasetSchemaVersion.V3: (
            COURT_SCHEMA_V3,
            (
                "canonical_court_dataset_v3",
                "canonical_court_orbit_plan_v3",
                "canonical_court_sample_v3",
                "court_renderer_semantic_manifest_v3",
                "court_dataset_performance_v4",
                "court_render_shard_attempt_v3",
            ),
            14,
            1,
        ),
        CourtDatasetSchemaVersion.V4: (
            COURT_SCHEMA_V4,
            (
                "canonical_court_dataset_v4",
                "canonical_court_safe_path_plan_v4",
                "canonical_court_sample_v4",
                "court_renderer_semantic_manifest_v4",
                "court_safe_trajectory_performance_v1",
                "court_render_shard_attempt_v4",
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
    assert COURT_SCHEMA_V3.semantic_class_names == COURT_KP_NAMES[:14]
    assert COURT_SCHEMA_V4.semantic_class_names == COURT_KP_NAMES[:14]


def test_v4_registry_does_not_claim_a_legacy_next_version_name() -> None:
    with pytest.raises(ValueError, match="Unknown Court performance schema"):
        court_schema_from_performance_schema("court_dataset_performance_v5")


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
