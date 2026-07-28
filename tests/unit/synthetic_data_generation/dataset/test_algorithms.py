"""Tests for strict configuration-selectable algorithm registries."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.algorithms import (
    AlgorithmDefinition,
    AlgorithmRegistry,
)


def test_registry_resolves_exact_names_without_fallback() -> None:
    registry = AlgorithmRegistry(
        namespace="example",
        definitions=(
            AlgorithmDefinition(
                name="first",
                implementation=1,
                description="First implementation.",
            ),
            AlgorithmDefinition(
                name="second",
                implementation=2,
                description="Second implementation.",
            ),
        ),
    )

    assert registry.names() == ("first", "second")
    assert registry.resolve("second") == 2
    with pytest.raises(ValueError, match="available choices: first, second"):
        registry.resolve("missing")


def test_registry_rejects_duplicate_and_empty_definitions() -> None:
    definition = AlgorithmDefinition(
        name="same",
        implementation=object(),
        description="One implementation.",
    )
    with pytest.raises(ValueError, match="Duplicate algorithm"):
        AlgorithmRegistry(
            namespace="example",
            definitions=(definition, definition),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        AlgorithmRegistry[object](namespace="empty", definitions=())
