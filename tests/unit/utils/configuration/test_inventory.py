"""Current configuration audit policy and runtime boundary inventory tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.utils.configuration import (
    DEFAULT_AUDIT_INVENTORY,
    AuditInventory,
    AuditRule,
    BoundaryKind,
)
from src.utils.configuration.catalog import BOUNDARY_CONTRACTS
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES


def test_synthetic_inventory_has_only_the_canonical_production_clis() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain == "synthetic_data_generation"
    }
    expected = {
        "src.synthetic_data_generation.scripts.run_appearance_variant": (
            "synthetic.appearance_variant",
            "src.synthetic_data_generation.appearance.configuration.validate_appearance_boundary",
        ),
        "src.synthetic_data_generation.scripts.compact_court_storage": (
            "synthetic.court_storage_compaction",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        ),
        "src.synthetic_data_generation.scripts.court_scene_assets": (
            "synthetic.court_scene_assets",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        ),
        "src.synthetic_data_generation.scripts.generate_court_scale": (
            "synthetic.court_scale_generation",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        ),
        "src.synthetic_data_generation.scripts.generate_publication_visualizations": (
            "synthetic.publication_visualization",
            "src.synthetic_data_generation.visualization.publication.configuration."
            "validate_publication_boundary",
        ),
        "src.synthetic_data_generation.scripts.review_court_dataset": (
            "synthetic.court_review",
            "src.utils.configuration.paths.NonHydraPathBoundary.validate",
        ),
        "src.synthetic_data_generation.scripts.run_scene_pipeline": (
            "synthetic.scene_pipeline",
            "src.synthetic_data_generation.configuration."
            "validate_scene_pipeline_boundary",
        ),
        "src.synthetic_data_generation.scripts.visualize_dataset": (
            "synthetic.dataset_visualization",
            "src.synthetic_data_generation.visualization.configuration."
            "validate_dataset_visualization_boundary",
        ),
        "src.synthetic_data_generation.scripts.visualize_scene_reveal": (
            "synthetic.scene_reveal",
            "src.synthetic_data_generation.visualization.reveal.configuration."
            "validate_scene_reveal_boundary",
        ),
    }

    assert set(boundaries) == set(expected)
    for module, (validator_key, validator_callable) in expected.items():
        assert boundaries[module].validator_key == validator_key
        assert boundaries[module].validator_callable == validator_callable


def test_manual_court_editor_has_an_explicit_non_hydra_path_boundary() -> None:
    boundary = next(
        boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.module == "src.synthetic_data_generation.scripts.edit_alignment"
    )

    assert boundary.domain == "manual_court_alignment"
    assert boundary.validator_key == "synthetic.manual_court_alignment"
    assert boundary.validator_callable == (
        "src.utils.configuration.paths.NonHydraPathBoundary.validate"
    )
    assert boundary.executable_module


def test_smpl_asset_installer_has_an_explicit_non_hydra_path_boundary() -> None:
    boundary = next(
        boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.module == "src.submodules.scripts.install_smpl_assets"
    )

    assert boundary.domain == "submodules"
    assert boundary.callable_name == "main"
    assert boundary.kind is BoundaryKind.ARGPARSE
    assert boundary.validator_key == "submodules.install_smpl_assets"
    assert boundary.validator_callable == (
        "src.utils.configuration.paths.NonHydraPathBoundary.validate"
    )
    assert boundary.executable_module


def test_task_local_generation_and_visualization_boundaries_remain_in_inventory() -> (
    None
):
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
        "src.tasks.blcs.scripts.generate_dataset_samples": (
            "blcs.generate_dataset_samples",
            "src.tasks.blcs.generate_dataset.samples.validate_dataset_samples_boundary",
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
        "src.tasks.plcs.scripts.generate_dataset_samples": (
            "plcs.generate_dataset_samples",
            "src.tasks.plcs.generate_dataset.samples.validate_dataset_samples_boundary",
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


def test_mixed_court_training_boundary_remains_explicitly_registered() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain == "court_detection"
    }
    boundary = boundaries["src.tasks.court_detection.scripts.train"]

    assert boundary.validator_key == "court_detection.train"
    assert boundary.validator_callable == (
        "src.tasks.court_detection.training.runner.validate_train_boundary"
    )
    assert boundary.executable_module


def test_court_learning_presentation_has_an_explicit_hydra_boundary() -> None:
    boundary = next(
        boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.module
        == "src.tasks.court_detection.scripts.visualize_learning_stages"
    )
    assert boundary.validator_key == "court_detection.learning_stages"
    assert boundary.validator_callable == (
        "src.tasks.court_detection.visualization.presentation.configuration."
        "validate_learning_stages_boundary"
    )
    contract = next(
        contract
        for contract in BOUNDARY_CONTRACTS
        if contract.boundary_id == f"{boundary.module}:main"
    )
    assert contract.validator_callable == boundary.validator_callable
    assert any(field.endswith(".run_bundle") for field in contract.field_paths)


def test_default_inventory_contains_only_current_policy() -> None:
    assert DEFAULT_AUDIT_INVENTORY.rules == tuple(AuditRule)
    assert DEFAULT_AUDIT_INVENTORY.boundaries == EXPECTED_RUNTIME_BOUNDARIES
    assert not hasattr(DEFAULT_AUDIT_INVENTORY, "migrations")
    assert not hasattr(DEFAULT_AUDIT_INVENTORY, "exemptions")


def test_inventory_rejects_duplicate_boundaries() -> None:
    boundary = DEFAULT_AUDIT_INVENTORY.boundaries[0]

    with pytest.raises(ValueError, match="must be unique"):
        AuditInventory(boundaries=(boundary, replace(boundary)))


def test_inventory_rejects_incomplete_validator_binding() -> None:
    boundary = DEFAULT_AUDIT_INVENTORY.boundaries[0]
    invalid = replace(boundary, validator_key=None)

    with pytest.raises(ValueError, match="complete binding"):
        AuditInventory(boundaries=(invalid,))
