"""Truthful LIVE/MIGRATED/EXEMPTED configuration-route inventory."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from src.utils.configuration import (
    DEFAULT_AUDIT_INVENTORY,
    AuditExemption,
    AuditInventory,
    AuditRule,
    MigrationAuthorityKind,
    MigrationStatus,
)
from src.utils.configuration.audit import (
    audit_source,
    inspect_source,
    regenerate_exemption_rows,
    regenerate_migration_rows,
    write_generated_inventory_data,
)
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES
from src.utils.configuration.source_oracle import (
    OracleCategory,
    inspect_raw_source,
)
from src.utils.paths import PROJECT_ROOT


def test_synthetic_inventory_has_only_the_canonical_scene_cli() -> None:
    boundaries = tuple(
        boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain == "synthetic_data_generation"
    )

    assert len(boundaries) == 1
    assert boundaries[0].module == (
        "src.synthetic_data_generation.scripts.run_scene_pipeline"
    )
    assert boundaries[0].validator_key == "synthetic.scene_pipeline"
    assert boundaries[0].validator_callable == (
        "src.synthetic_data_generation.configuration.validate_scene_pipeline_boundary"
    )


def test_task_local_generation_and_visualization_boundaries_remain_in_inventory() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain in {"blcs", "plcs"}
    }
    expected = {
        "src.tasks.blcs.scripts.generate_dataset": (
            "blcs.generate_dataset",
            "src.tasks.blcs.configuration.validate_generation_boundary",
        ),
        "src.tasks.blcs.scripts.preview_augmentation": (
            "blcs.preview_augmentation",
            "src.tasks.blcs.configuration.validate_preview_boundary",
        ),
        "src.tasks.blcs.scripts.visualize": (
            "blcs.visualize",
            "src.tasks.blcs.configuration.validate_visualization_boundary",
        ),
        "src.tasks.plcs.scripts.generate_dataset": (
            "plcs.generate_dataset",
            "src.tasks.plcs.generate_dataset.config._validate_boundary",
        ),
        "src.tasks.plcs.scripts.preview_augmentation": (
            "plcs.preview_augmentation",
            "src.tasks.plcs.configuration._validate_preview_boundary",
        ),
        "src.tasks.plcs.scripts.visualize": (
            "plcs.visualize",
            "src.tasks.plcs.configuration._validate_visualization_boundary",
        ),
    }

    for module, (validator_key, validator_callable) in expected.items():
        boundary = boundaries[module]
        assert boundary.validator_key == validator_key
        assert boundary.validator_callable == validator_callable


def test_inventory_contains_disjoint_truthful_route_states() -> None:
    counts = Counter(record.status for record in DEFAULT_AUDIT_INVENTORY.migrations)

    assert set(counts) == set(MigrationStatus)
    assert all(
        record.expected_current_occurrences == 0
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.status is MigrationStatus.MIGRATED
    )
    assert all(
        record.expected_current_occurrences > 0
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.status in {MigrationStatus.LIVE, MigrationStatus.EXEMPTED}
    )


def test_surviving_route_cannot_be_declared_migrated() -> None:
    live = next(
        record
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.status is MigrationStatus.LIVE
    )
    invalid = replace(live, status=MigrationStatus.MIGRATED)

    with pytest.raises(ValueError, match="truthful status"):
        AuditInventory(
            boundaries=DEFAULT_AUDIT_INVENTORY.boundaries,
            migrations=(invalid,),
            exemptions=DEFAULT_AUDIT_INVENTORY.exemptions,
            rules=DEFAULT_AUDIT_INVENTORY.rules,
        )


def test_configured_path_join_exemption_requires_exempted_status() -> None:
    record = next(
        record
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.former_module == "src.tasks.base.training.runner"
        and record.former_qualified_name == "BaseTrainingRunner.build_callbacks"
        and record.former_route.startswith("configured-path-join:")
    )
    exemption = next(
        exemption
        for exemption in DEFAULT_AUDIT_INVENTORY.exemptions
        if exemption.module == record.former_module
        and exemption.qualified_name == record.former_qualified_name
        and exemption.line == record.former_line
        and exemption.rule is AuditRule.PATH_JOIN
    )

    with pytest.raises(ValueError, match="EXEMPTED routes"):
        AuditInventory(
            boundaries=DEFAULT_AUDIT_INVENTORY.boundaries,
            migrations=(replace(record, status=MigrationStatus.LIVE),),
            exemptions=(exemption,),
            rules=DEFAULT_AUDIT_INVENTORY.rules,
        )


def test_vitpose_fallback_history_is_migrated_but_typed_access_is_live() -> None:
    records = tuple(
        record
        for record in DEFAULT_AUDIT_INVENTORY.migrations
        if record.former_module.endswith("vitpose.heatmap_head")
    )

    assert any(
        record.status is MigrationStatus.MIGRATED
        and "extra.get('num_conv_layers'" in record.former_route
        for record in records
    )
    assert any(
        record.status is MigrationStatus.LIVE
        and "config.num_conv_kernels" in record.former_route
        for record in records
    )


def test_checked_in_authorities_and_occurrences_match_source() -> None:
    report = inspect_source((PROJECT_ROOT / "src").resolve())

    assert report.passed


def test_new_source_route_is_reported_as_unrecorded(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "new_route.py").write_text(
        "def run(config: dict[str, object]) -> object:\n"
        '    return config["new_required"]\n',
        encoding="utf-8",
    )

    report = inspect_source(source_root, inventory=DEFAULT_AUDIT_INVENTORY)

    assert any(
        issue.record_id == "<unrecorded-route>" for issue in report.migration_issues
    )


def test_generated_inventory_writer_is_deterministic_and_scoped(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    target = source_root / "utils" / "configuration"
    target.mkdir(parents=True)

    first_counts = write_generated_inventory_data(
        source_root,
        source_revision="test-revision",
    )
    first_migration = (target / "migration_data.py").read_bytes()
    first_exemption = (target / "exemption_data.py").read_bytes()
    second_counts = write_generated_inventory_data(
        source_root,
        source_revision="test-revision",
    )

    assert first_counts == second_counts
    assert (target / "migration_data.py").read_bytes() == first_migration
    assert (target / "exemption_data.py").read_bytes() == first_exemption
    assert b"MIGRATION_SOURCE_REVISION = 'test-revision'" in first_migration


def test_regeneration_accepts_only_an_exact_reviewed_exemption(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "reviewed.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def build(root: Path) -> Path:\n"
        "    return root / 'tool'\n",
        encoding="utf-8",
    )
    approval = AuditExemption.classified(
        module="src.reviewed",
        qualified_name="build",
        line=4,
        rule=AuditRule.PATH_JOIN,
        reason_code="strict-schema",
    )

    rows, exemptions, unresolved = regenerate_exemption_rows(
        source_root,
        approved_exemptions=(approval,),
    )

    assert not unresolved
    assert rows == (("src.reviewed", "build", 4, "path-join", "strict-schema"),)
    assert exemptions == (approval,)
    with pytest.raises(ValueError, match="do not match current findings"):
        regenerate_exemption_rows(
            source_root,
            approved_exemptions=(replace(approval, line=5),),
        )


def test_regeneration_discovers_semantic_mapping_and_path_routes() -> None:
    rows = regenerate_migration_rows((PROJECT_ROOT / "src").resolve())
    live_routes = {
        (str(row[1]), str(row[5]))
        for row in rows
        if row[8] in {"live", "exempted"}
    }

    expected = (
        ("src.tasks.blcs.configuration", "model['hidden_dim']"),
        ("src.tasks.slcs.configuration", "raw['window_size']"),
        ("src.tasks.base.configuration", "mapping[key]"),
        ("src.utils.io", "Path(path)"),
        (
            "src.synthetic_data_generation.dataset.blcs.assembler",
            "output_directory / 'dataset.json'",
        ),
    )
    for module, route_fragment in expected:
        assert any(
            candidate_module == module and route_fragment in route
            for candidate_module, route in live_routes
        )

    assert not any(
        module.endswith("vitpose.kp2d_utils")
        and "scale[0] / (output_size[0] - 1.0)" in route
        for module, route in live_routes
    )
    assert not any(
        module.endswith("alignment.scene_provider.geometry_bridge")
        and (
            "1.0 / (1.0 + cosine)" in route
            or "1.0 / np.median" in route
        )
        for module, route in live_routes
    )
    assert not any(
        str(row[1]).endswith("alignment.scene_provider.geometry_bridge")
        and (
            "1.0 / (1.0 + cosine)" in str(row[5])
            or "1.0 / np.median" in str(row[5])
        )
        for row in rows
    )


def test_independent_oracle_finds_mapping_and_verified_path_dataflow(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "sample.py").write_text(
        "from collections.abc import Mapping\n"
        "from pathlib import Path\n"
        "def read_config(raw: Mapping[str, object], root: Path) -> object:\n"
        "    configured = raw['required']\n"
        "    output = root / 'child'\n"
        "    return configured, output\n",
        encoding="utf-8",
    )

    occurrences = inspect_raw_source(source_root)

    assert any(
        occurrence.category is OracleCategory.CONFIGURATION_REFERENCE
        for occurrence in occurrences
    )
    assert any(
        occurrence.category is OracleCategory.PATH_RESOLUTION
        for occurrence in occurrences
    )


def test_regeneration_tracks_typed_configuration_casts(tmp_path: Path) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "sample.py").write_text(
        "from typing import cast\n"
        "class FeatureConfig:\n"
        "    enabled: bool\n"
        "def enabled(runtime: FeatureConfig) -> bool:\n"
        "    return cast(FeatureConfig, runtime).enabled\n",
        encoding="utf-8",
    )

    rows = regenerate_migration_rows(source_root)

    assert any(
        row[1] == "src.sample"
        and row[2] == "enabled"
        and row[7] == "configuration-reference"
        and "cast(FeatureConfig, runtime).enabled" in str(row[5])
        for row in rows
    )


def test_independent_oracle_rejects_numeric_division_path_false_positive(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "normalization.py").write_text(
        "def normalize(cosine: float, scale_path: float) -> float:\n"
        "    rotation = 1.0 / (1.0 + cosine)\n"
        "    return rotation / scale_path\n",
        encoding="utf-8",
    )

    occurrences = inspect_raw_source(source_root)

    assert not any(
        occurrence.category is OracleCategory.PATH_RESOLUTION
        for occurrence in occurrences
    )


def test_classifiers_distinguish_logical_or_from_configuration_fallback(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    source_root.mkdir()
    (source_root / "validation.py").write_text(
        "from collections.abc import Mapping\n"
        "def validate_value(value: object) -> None:\n"
        "    if type(value) is not str or not value:\n"
        "        raise TypeError\n"
        "def read_config(config: Mapping[str, object]) -> object:\n"
        "    return config['required'] or 'fallback'\n"
        "def normalize_optional(config: Mapping[str, object]) -> str | None:\n"
        "    value = config['optional']\n"
        "    return None if value is None else str(value)\n"
        "_CACHE: object | None = None\n"
        "def cached(config: Mapping[str, object]) -> object:\n"
        "    global _CACHE\n"
        "    if _CACHE is None:\n"
        "        _CACHE = config['required']\n"
        "    return _CACHE\n",
        encoding="utf-8",
    )

    occurrences = inspect_raw_source(source_root)
    findings = audit_source(source_root)

    runtime_defaults = {
        occurrence.line
        for occurrence in occurrences
        if occurrence.category is OracleCategory.PYTHON_RUNTIME_DEFAULT
    }
    null_coalescing = {
        finding.line
        for finding in findings
        if finding.rule is AuditRule.NULL_COALESCING
    }
    assert runtime_defaults == {6}
    assert null_coalescing == {6}


def test_actual_geometry_math_has_no_path_occurrences() -> None:
    occurrences = inspect_raw_source((PROJECT_ROOT / "src").resolve())

    assert not any(
        occurrence.module.endswith(
            "alignment.scene_provider.geometry_bridge"
        )
        and occurrence.qualified_name
        in {"_similarity_from_cameras", "_transform_cameras"}
        and occurrence.category is OracleCategory.PATH_RESOLUTION
        for occurrence in occurrences
    )


def test_live_path_authority_matches_actual_route() -> None:
    source_root = (PROJECT_ROOT / "src").resolve()
    _, exemptions, unresolved = regenerate_exemption_rows(source_root)
    assert not unresolved
    rows = regenerate_migration_rows(source_root, exemptions=exemptions)

    for row in rows:
        if row[7] != "path-resolution" or row[8] not in {"live", "exempted"}:
            continue
        route = str(row[5])
        if row[9] == "path-resolver":
            assert row[10] == "src.utils.configuration.paths.PathResolver.resolve"
            assert "resolver" in route or "yaml-source-route" in route
        elif row[9] == MigrationAuthorityKind.SCHEMA_FIELD.value:
            assert str(row[10]).endswith((".PATH_BOUNDARY", ".RuntimePathRoots"))
            assert row[11]
        else:
            expected = str(row[1]) if row[2] == "<module>" else f"{row[1]}.{row[2]}"
            assert row[9] == "execution-input"
            assert row[10] == expected
