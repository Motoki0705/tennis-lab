"""Negative tests for intentionally removed PLCS compatibility paths."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.tasks.plcs.utils.pose_geometry",
        "src.tasks.plcs.validation_matrix",
        "src.tasks.plcs.visualization.adapters.predict_inputs",
        "src.tasks.plcs.data.chunk_manager",
    ],
)
def test_removed_module_has_no_forwarding_path(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    ("module", "symbol"),
    [
        ("src.tasks.plcs", "SceneGenerator"),
        ("src.tasks.plcs.models", "PLCSModel"),
        ("src.tasks.plcs.data", "adapt_batch_for_model_profile"),
        ("src.tasks.plcs.data", "SceneGenerator"),
        ("src.tasks.plcs.training", "position_loss"),
        ("src.tasks.plcs.training", "rotation_loss"),
    ],
)
def test_removed_compatibility_symbol_is_not_reexported(
    module: str, symbol: str
) -> None:
    imported = importlib.import_module(module)
    assert not hasattr(imported, symbol)
