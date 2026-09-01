"""Current configuration audit policy and runtime boundary inventory tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.utils.configuration import (
    DEFAULT_AUDIT_INVENTORY,
    AuditInventory,
    AuditRule,
)
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES


def test_synthetic_inventory_has_only_the_canonical_production_clis() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain == "synthetic_data_generation"
    }
    expected = {
        "src.synthetic_data_generation.scripts.generate_publication_visualizations": (
            "synthetic.publication_visualization",
            "src.synthetic_data_generation.visualization.publication.configuration."
            "validate_publication_boundary",
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
    }

    assert set(boundaries) == set(expected)
    for module, (validator_key, validator_callable) in expected.items():
        assert boundaries[module].validator_key == validator_key
        assert boundaries[module].validator_callable == validator_callable


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
    boundary = boundaries["src.tasks.court_detection.scripts.train_mixed"]

    assert boundary.validator_key == "court_detection.train_mixed"
    assert boundary.validator_callable == (
        "src.tasks.court_detection.training.runner_mixed.validate_mixed_train_boundary"
    )
    assert boundary.executable_module


def test_court_alignment_train_and_evaluate_boundaries_are_registered() -> None:
    boundaries = {
        boundary.module: boundary
        for boundary in EXPECTED_RUNTIME_BOUNDARIES
        if boundary.domain == "court_alignment"
    }
    assert {
        module: (boundary.validator_key, boundary.validator_callable)
        for module, boundary in boundaries.items()
    } == {
        "src.tasks.court_alignment.scripts.train": (
            "court_alignment.train",
            "src.tasks.court_alignment.configuration.validate_training_boundary",
        ),
        "src.tasks.court_alignment.scripts.evaluate": (
            "court_alignment.evaluate",
            "src.tasks.court_alignment.configuration.validate_evaluation_boundary",
        ),
    }


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
