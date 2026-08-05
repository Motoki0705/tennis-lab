"""Unit tests for the canonical shared visualization configuration contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.base.configuration import SceneVisualizationConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

pytestmark = pytest.mark.unit


def _resolver(tmp_path: Path) -> PathResolver:
    roots = RuntimePathRoots.from_mapping(
        {
            "project_root": ".",
            "data_root": "data",
            "checkpoint_root": "checkpoints",
            "artifact_root": "artifacts",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "external",
        },
        repository_root=tmp_path,
    )
    return PathResolver(roots)


def _visualization_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "mode": "prediction",
        "scene_path": "scenes/scene_0001",
        "checkpoint": None,
        "device": "cpu",
        "animation_view": "3d",
        "fps": 30.0,
        "save": None,
        "camera": 0,
        "cameras": None,
        "info": False,
        "style": {},
        "view_3d": {},
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("all", "all"),
        ("0,1,2", (0, 1, 2)),
        ([0, 2, 5], (0, 2, 5)),
        ((1, 3), (1, 3)),
    ],
)
def test_camera_selection_uses_typed_canonical_contract(
    tmp_path: Path,
    raw: object,
    expected: tuple[int, ...] | str | None,
) -> None:
    parsed = SceneVisualizationConfig.from_mapping(
        _visualization_config(cameras=raw), resolver=_resolver(tmp_path)
    )
    assert parsed.cameras == expected


def test_invalid_camera_string_raises(tmp_path: Path) -> None:
    with pytest.raises(SemanticConfigurationError, match="comma-separated integers"):
        SceneVisualizationConfig.from_mapping(
            _visualization_config(cameras="a,b"), resolver=_resolver(tmp_path)
        )


def test_camera_list_rejects_non_exact_int(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationTypeError, match="exact int"):
        SceneVisualizationConfig.from_mapping(
            _visualization_config(cameras=[0, True]), resolver=_resolver(tmp_path)
        )


def test_missing_camera_key_raises(tmp_path: Path) -> None:
    config = _visualization_config()
    del config["cameras"]
    with pytest.raises(MissingConfigurationKeyError, match="visualization.cameras"):
        SceneVisualizationConfig.from_mapping(config, resolver=_resolver(tmp_path))


def test_removed_or_unknown_key_raises(tmp_path: Path) -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="visualization.camera_ids"):
        SceneVisualizationConfig.from_mapping(
            _visualization_config(camera_ids=[0, 1]), resolver=_resolver(tmp_path)
        )
