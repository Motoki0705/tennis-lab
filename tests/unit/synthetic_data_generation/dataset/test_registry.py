"""Tests for the central extensible dataset registry."""

from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.registry import (
    available_dataset_pipelines,
    get_dataset_pipeline,
)


def test_builtin_registry_owns_all_current_datasets() -> None:
    definitions = available_dataset_pipelines()

    assert tuple(item.name for item in definitions) == ("blcs", "court", "plcs")
    assert get_dataset_pipeline("blcs").dataset_name == "blcs"
    assert get_dataset_pipeline("court").dataset_name == "court"
    assert get_dataset_pipeline("plcs").dataset_name == "plcs"


def test_registry_rejects_unknown_dataset_without_fallback() -> None:
    with pytest.raises(ValueError, match="available choices: blcs, court, plcs"):
        get_dataset_pipeline("unknown")
