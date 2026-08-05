"""Source-complete inspectable configuration-contract catalog tests."""

from __future__ import annotations

import importlib

import pytest

from src.configuration_contracts import (
    ADAPTER_CONTRACTS,
    BOUNDARY_CONTRACTS,
    SOURCE_CONTRACT_DECLARATIONS,
)
from src.utils.configuration import (
    ConfigurationAbsencePolicy,
    ConfigurationDefaultPolicy,
    ConfigurationPrecedence,
    ConfigurationTypeError,
    StrictConfigSchema,
    discover_configuration_contracts,
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


def test_all_84_runtime_boundaries_expose_truthful_authorities() -> None:
    assert len(BOUNDARY_CONTRACTS) == 84
    assert len({contract.boundary_id for contract in BOUNDARY_CONTRACTS}) == 84
    for contract in BOUNDARY_CONTRACTS:
        assert contract.validator_callable
        assert contract.authority_symbols
        assert contract.semantic_constraint_authorities
        assert all("source-validated-input" not in path for path in contract.field_paths)


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
            if contract.boundary_id
            == f"src.tasks.slcs.scripts.{script}:main"
        )
        schema_symbols = {
            symbol
            for symbol in boundary.authority_symbols
            if symbol.endswith("_BOUNDARY_SCHEMA")
        }
        assert schema_symbols == {
            f"src.tasks.slcs.configuration.{schema_name}"
        }
        assert boundary.path_role_authorities


def test_synthetic_registry_binds_each_boundary_to_one_top_level_schema() -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == "src.synthetic_data_generation.scripts.dataset.run_pipeline:main"
    )

    top_level = {
        symbol
        for symbol in boundary.authority_symbols
        if symbol.startswith("src.synthetic_data_generation.configuration.")
        and symbol.endswith("SCHEMA")
        and symbol.split(".")[-1]
        in {
            "PIPELINE_SCHEMA",
            "FEATURE_FIT_SCHEMA",
            "INFER_SCHEMA",
            "FIT_GROUND_SCHEMA",
            "CALIBRATE_SCHEMA",
            "EXPORT_SCHEMA",
            "GEOMETRY_BRIDGE_SCHEMA",
            "VALIDATION_MATRIX_SCHEMA",
        }
    }
    assert top_level == {
        "src.synthetic_data_generation.configuration.PIPELINE_SCHEMA"
    }


def test_non_hydra_boundary_catalog_exposes_exact_path_roles() -> None:
    boundary = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id
        == (
            "src.synthetic_data_generation.dataset.blcs.components."
            "asset_preparation:main"
        )
    )

    assert any(
        authority.endswith(".PATH_BOUNDARY")
        for authority in boundary.authority_symbols
    )
    assert any("asset_spec:path-role:external_asset" in value for value in boundary.path_role_authorities)
    assert any("output_dir:path-role:artifact" in value for value in boundary.path_role_authorities)


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
        contract.boundary_id.startswith(domain)
        for contract in BOUNDARY_CONTRACTS
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
        "src.synthetic_data_generation.configuration.SyntheticRuntimeConfig",
        "src.submodules.configuration.SubmoduleRuntimeConfig",
    ],
)
def test_representative_domain_adapter_is_source_discovered(symbol: str) -> None:
    contract = next(
        candidate for candidate in ADAPTER_CONTRACTS if candidate.adapter_symbol == symbol
    )

    assert contract.authority_kind == "typed-dataclass"
    assert contract.fields
    assert all(field.required for field in contract.fields)
