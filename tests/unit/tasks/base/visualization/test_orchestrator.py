"""Unit tests for the canonical shared visualization configuration contract."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tasks.base.configuration import SceneVisualizationConfig
from src.tasks.base.visualization.orchestrator import (
    parse_float_triplet,
    parse_hw,
    parse_rgb,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathResolver,
    RuntimePathRoots,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

pytestmark = pytest.mark.unit


def _parse_hw(value: object) -> object:
    return parse_hw(value, name="value")


def _parse_rgb(value: object) -> object:
    return parse_rgb(value, name="value")


def _parse_float_triplet(value: object) -> object:
    return parse_float_triplet(value, name="value")


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


def test_triplet_parsers_accept_hydra_list_config_without_weakening_types() -> None:
    config = OmegaConf.create(
        {
            "image_size": [288, 512],
            "rgb": [12, 34, 255],
            "normalization": [0.485, 0.456, 0.406],
        }
    )

    assert parse_hw(config.image_size, name="image_size") == (288, 512)
    assert parse_rgb(config.rgb, name="rgb") == (12, 34, 255)
    assert parse_float_triplet(config.normalization, name="normalization") == (
        0.485,
        0.456,
        0.406,
    )


@pytest.mark.parametrize(
    "value",
    [
        "1,2,3",
        b"123",
        bytearray(b"123"),
        {"first": 1, "second": 2},
        {1, 2, 3},
        iter((1, 2, 3)),
    ],
)
@pytest.mark.parametrize(
    "parser",
    [_parse_hw, _parse_rgb, _parse_float_triplet],
)
def test_triplet_parsers_reject_non_sequence_or_ambiguous_containers(
    parser: Callable[[object], object],
    value: object,
) -> None:
    with pytest.raises(TypeError, match="non-string sequence"):
        parser(value)


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (_parse_hw, [1]),
        (_parse_hw, [1, True]),
        (_parse_hw, [1, 2.0]),
        (_parse_rgb, [1, 2]),
        (_parse_rgb, [1, 2, True]),
        (_parse_rgb, [1, 2, 3.0]),
        (_parse_float_triplet, [0.1, 0.2]),
        (_parse_float_triplet, [0.1, 0.2, True]),
        (_parse_float_triplet, [0.1, 0.2, "0.3"]),
    ],
)
def test_triplet_parsers_reject_wrong_length_and_non_exact_element_types(
    parser: Callable[[object], object],
    value: object,
) -> None:
    with pytest.raises(TypeError):
        parser(value)


@pytest.mark.parametrize("rgb", [(-1, 0, 0), (0, 0, 256)])
def test_rgb_parser_rejects_out_of_range_channels(
    rgb: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 255\]"):
        parse_rgb(rgb, name="rgb")


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
