"""Source-complete inspectable configuration-contract catalog tests."""

from __future__ import annotations

import importlib
from importlib.machinery import PathFinder

import pytest

from src.utils.configuration import (
    DEFAULT_AUDIT_INVENTORY,
    ConfigurationAbsencePolicy,
    ConfigurationDefaultPolicy,
    ConfigurationPrecedence,
    ConfigurationTypeError,
    StrictConfigSchema,
    discover_configuration_contracts,
)
from src.utils.configuration.catalog import (
    ADAPTER_CONTRACTS,
    BOUNDARY_CONTRACTS,
    SOURCE_CONTRACT_DECLARATIONS,
)
from src.utils.paths import PROJECT_ROOT


def test_catalog_has_exact_source_declaration_parity() -> None:
    symbols = {contract.adapter_symbol for contract in ADAPTER_CONTRACTS}

    assert SOURCE_CONTRACT_DECLARATIONS.all_symbols <= symbols
    assert len(symbols) == len(ADAPTER_CONTRACTS)
    assert (
        discover_configuration_contracts((PROJECT_ROOT / "src").resolve())
        == ADAPTER_CONTRACTS
    )
    assert all(contract.fields for contract in ADAPTER_CONTRACTS)


@pytest.mark.parametrize(
    "old_module",
    [
        "src.configuration_contracts",
        "src.configuration_validation",
        "src.utils.configuration.validation",
    ],
)
def test_removed_configuration_modules_do_not_redirect(old_module: str) -> None:
    parent = PROJECT_ROOT.joinpath(*old_module.split(".")[:-1])

    assert PathFinder.find_spec(old_module, [str(parent)]) is None


@pytest.mark.parametrize(
    "domain",
    [
        "src.tasks.base",
        "src.tasks.ball_detection",
        "src.tasks.blcs",
        "src.tasks.court_detection",
        "src.tasks.plcs",
        "src.tasks.slcs",
        "src.tennis_scene",
        "src.synthetic_data_generation",
        "src.submodules",
        "src.utils.configuration.operations",
    ],
)
def test_every_runtime_domain_has_required_inspectable_authorities(
    domain: str,
) -> None:
    contracts = tuple(
        contract
        for contract in ADAPTER_CONTRACTS
        if contract.adapter_symbol.startswith(domain)
    )

    assert contracts, domain
    assert any(contract.authority_kind == "typed-dataclass" for contract in contracts)
    assert any(field.required for contract in contracts for field in contract.fields)


def test_adapter_fields_have_truthful_default_precedence_and_absence_policy() -> None:
    for contract in ADAPTER_CONTRACTS:
        for field in contract.inspect():
            assert field.expected_types
            assert field.value_constraints
            assert field.default_policy is ConfigurationDefaultPolicy.COMPOSITION_OWNED
            assert (
                field.precedence_authority
                is ConfigurationPrecedence.COMPOSED_VALUE_ONLY
            )
            if field.required:
                assert field.absence_policy is ConfigurationAbsencePolicy.REQUIRED
            else:
                assert field.absence_policy in {
                    ConfigurationAbsencePolicy.OPTIONAL_OMITTED,
                    ConfigurationAbsencePolicy.OPTIONAL_AS_NONE,
                }


def test_court_v4_policy_fields_are_required_composition_authority() -> None:
    expected_fields = {
        "src.synthetic_data_generation.configuration.CourtTrajectoryPolicyV4": {
            "anchored_half_width_m",
            "anchored_half_height_m",
            "anchored_corner_radius_m",
            "anchored_raised_lift_m",
            "anchored_reference_point_count",
        },
        "src.synthetic_data_generation.configuration.CourtDatasetConfiguration": {
            "support",
            "benchmark_decision_id",
            "required_coverage",
        },
    }

    for symbol, expected in expected_fields.items():
        contract = next(
            candidate
            for candidate in ADAPTER_CONTRACTS
            if candidate.adapter_symbol == symbol
        )
        fields = {
            field.path.rpartition(".")[2]: field
            for field in contract.fields
            if field.path.rpartition(".")[2] in expected
        }

        assert set(fields) == expected
        assert all(field.required for field in fields.values())
        assert all(
            field.default_policy is ConfigurationDefaultPolicy.COMPOSITION_OWNED
            for field in fields.values()
        )
        assert all(
            field.absence_policy is ConfigurationAbsencePolicy.REQUIRED
            for field in fields.values()
        )


def test_operation_build_json_is_optional_and_absence_maps_to_none() -> None:
    contract = next(
        contract
        for contract in ADAPTER_CONTRACTS
        if contract.adapter_symbol
        == "src.utils.configuration.operations.OperationEnvironmentConfig"
    )
    field = next(
        candidate
        for candidate in contract.fields
        if candidate.path.endswith(".dino_ops_build_config")
    )

    assert not field.required
    assert field.absence_policy is ConfigurationAbsencePolicy.OPTIONAL_AS_NONE


def test_all_inventoried_runtime_boundaries_expose_truthful_authorities() -> None:
    boundary_ids = {contract.boundary_id for contract in BOUNDARY_CONTRACTS}
    expected_ids = {
        f"{boundary.module}:{boundary.callable_name}"
        for boundary in DEFAULT_AUDIT_INVENTORY.boundaries
    }

    assert boundary_ids == expected_ids
    assert len(boundary_ids) == len(BOUNDARY_CONTRACTS)
    for contract in BOUNDARY_CONTRACTS:
        assert contract.validator_callable
        assert contract.authority_symbols
        assert contract.semantic_constraint_authorities
        assert all(
            "source-validated-input" not in path for path in contract.field_paths
        )


def test_slcs_boundaries_bind_only_their_actual_public_boundary_schema() -> None:
    expected = {
        "analyze_predictions": "SLCS_ANALYSIS_BOUNDARY_SCHEMA",
        "evaluate": "SLCS_EVALUATION_BOUNDARY_SCHEMA",
        "make_splits": "SLCS_SPLITS_BOUNDARY_SCHEMA",
        "precompute_dino_tokens": "SLCS_PRECOMPUTE_BOUNDARY_SCHEMA",
        "predict_clip": "SLCS_PREDICTION_BOUNDARY_SCHEMA",
        "train": "SLCS_TRAINING_BOUNDARY_SCHEMA",
    }

    for script, schema_name in expected.items():
        boundary = next(
            contract
            for contract in BOUNDARY_CONTRACTS
            if contract.boundary_id == f"src.tasks.slcs.scripts.{script}:main"
        )
        schema_symbols = {
            symbol
            for symbol in boundary.authority_symbols
            if symbol.endswith("_BOUNDARY_SCHEMA")
        }
        assert schema_symbols == {f"src.tasks.slcs.configuration.{schema_name}"}
        assert boundary.path_role_authorities


def test_synthetic_registry_exposes_only_the_canonical_production_boundaries() -> None:
    publication_boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.generate_publication_visualizations:main"
    )
    benchmark_boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety:main"
    )
    scene_boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.run_scene_pipeline:main"
    )
    visualization_boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.visualize_dataset:main"
    )

    synthetic_boundaries = {
        contract.boundary_id
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id.startswith("src.synthetic_data_generation")
    }
    assert synthetic_boundaries == {
        "src.synthetic_data_generation.scripts.generate_publication_visualizations:main",
        "src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety:main",
        "src.synthetic_data_generation.scripts.run_scene_pipeline:main",
        "src.synthetic_data_generation.scripts.visualize_dataset:main",
    }
    assert benchmark_boundary.validator_callable == (
        "src.synthetic_data_generation.scripts."
        "evaluate_court_trajectory_safety._validate_boundary"
    )
    assert (
        "src.synthetic_data_generation.scripts."
        "evaluate_court_trajectory_safety.BenchmarkConfiguration"
        in benchmark_boundary.authority_symbols
    )
    assert (
        "src.synthetic_data_generation.configuration.ScenePipelineConfiguration"
        in scene_boundary.authority_symbols
    )
    assert visualization_boundary.validator_callable == (
        "src.synthetic_data_generation.visualization.configuration."
        "validate_dataset_visualization_boundary"
    )
    assert any(
        authority.endswith(".build_visualization_request")
        for authority in visualization_boundary.semantic_constraint_authorities
    )
    assert publication_boundary.validator_callable == (
        "src.synthetic_data_generation.visualization.publication.configuration."
        "validate_publication_boundary"
    )


def test_synthetic_boundary_catalog_exposes_exact_canonical_path_roles() -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.run_scene_pipeline:main"
    )

    assert any("source_video" in value for value in boundary.path_role_authorities)
    assert any(
        "path-role:external_asset" in value for value in boundary.path_role_authorities
    )
    assert any("path-role:data" in value for value in boundary.path_role_authorities)
    assert not any(
        "asset_preparation" in value for value in boundary.path_role_authorities
    )


@pytest.mark.parametrize(
    "boundary_id,task_prefix",
    [
        (
            "src.tasks.blcs.scripts.generate_dataset:main",
            "src.tasks.blcs.",
        ),
        (
            "src.tasks.blcs.scripts.visualize:main",
            "src.tasks.blcs.",
        ),
        (
            "src.tasks.plcs.scripts.generate_dataset:main",
            "src.tasks.plcs.",
        ),
        (
            "src.tasks.plcs.scripts.visualize:main",
            "src.tasks.plcs.",
        ),
    ],
)
def test_task_local_boundaries_keep_task_local_configuration_authority(
    boundary_id: str,
    task_prefix: str,
) -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id == boundary_id
    )

    assert boundary.authority_symbols
    assert all(
        symbol.startswith((task_prefix, "src.tasks.base.", "src.utils.configuration."))
        for symbol in boundary.authority_symbols
    )
    assert not any(
        symbol.startswith("src.synthetic_data_generation.")
        for symbol in boundary.authority_symbols
    )


def test_boundary_catalog_follows_package_reexport_to_actual_adapter() -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.tasks.court_detection.scripts.annotate_youtube_keypoints:main"
    )

    assert (
        "src.tasks.court_detection.generate_dataset.annotation_session."
        "AnnotationSessionConfig"
    ) in boundary.authority_symbols


@pytest.mark.parametrize(
    "domain",
    [
        "src.tasks.ball_detection",
        "src.tasks.blcs",
        "src.tasks.court_detection",
        "src.tasks.plcs",
        "src.tasks.slcs",
        "src.tennis_scene",
        "src.synthetic_data_generation",
        "src.submodules",
    ],
)
def test_every_issue_domain_has_runtime_boundary_constraint_metadata(
    domain: str,
) -> None:
    assert any(
        contract.boundary_id.startswith(domain) for contract in BOUNDARY_CONTRACTS
    )


def test_every_strict_schema_field_rejects_an_invalid_exact_type() -> None:
    for symbol in SOURCE_CONTRACT_DECLARATIONS.schema_symbols:
        module_name, _, attribute = symbol.rpartition(".")
        schema = getattr(importlib.import_module(module_name), attribute)
        assert isinstance(schema, StrictConfigSchema)
        for name, field in schema.fields.items():
            with pytest.raises(ConfigurationTypeError):
                field.validate(object(), path=f"{schema.name}.{name}")


@pytest.mark.parametrize(
    "symbol",
    [
        "src.tasks.base.configuration.TrainingRuntimeConfig",
        "src.tasks.ball_detection.configuration.BallYoutubePathContract",
        "src.tasks.blcs.configuration.TrackQueryModelConfig",
        "src.tasks.court_detection.configuration.CourtTrainingConfig",
        "src.tasks.plcs.configuration.PLCSTrainingConfig",
        "src.tasks.slcs.configuration.SLCSTrainingRuntimeConfig",
        "src.tennis_scene.configuration.PipelineRuntimeConfig",
        "src.synthetic_data_generation.configuration.ScenePipelineConfiguration",
        "src.submodules.configuration.SubmoduleRuntimeConfig",
    ],
)
def test_representative_domain_adapter_is_source_discovered(symbol: str) -> None:
    contract = next(
        candidate
        for candidate in ADAPTER_CONTRACTS
        if candidate.adapter_symbol == symbol
    )

    assert contract.authority_kind == "typed-dataclass"
    assert contract.fields
    assert all(field.required for field in contract.fields)
