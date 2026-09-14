"""The PLCS visualization package resolves its heavy exports lazily."""

from __future__ import annotations

import importlib

import pytest

MODULE = "src.tasks.plcs.visualization"

PUBLIC_NAMES = (
    "PLCSSceneRenderer",
    "PoseRenderScene",
    "RuntimeConfig",
    "build_runtime_config",
    "run_visualization",
)


def test_public_exports_resolve_on_first_access() -> None:
    module = importlib.import_module(MODULE)
    assert sorted(module.__all__) == sorted(PUBLIC_NAMES)
    for name in PUBLIC_NAMES:
        assert getattr(module, name) is not None


def test_export_map_matches_the_public_names() -> None:
    module = importlib.import_module(MODULE)
    assert sorted(module._EXPORTS) == sorted(module.__all__)


def test_unknown_attribute_raises_attribute_error() -> None:
    module = importlib.import_module(MODULE)
    with pytest.raises(AttributeError):
        module.not_a_visualization_export  # noqa: B018
