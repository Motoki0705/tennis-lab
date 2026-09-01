"""Strict Hydra boundary for canonical dataset visualization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.visualization.contracts import (
    DEFAULT_COURT_OVERLAY_CONFIGURATION,
    CourtOverlayConfiguration,
    CourtOverlayMode,
    DatasetVisualizationDomain,
    DatasetVisualizationRequest,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

DATASET_VISUALIZATION_BOUNDARY = "synthetic.dataset_visualization"


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def _exact(
    value: object,
    *,
    name: str,
    keys: set[str],
    optional_keys: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    actual = set(result)
    if not keys.issubset(actual) or not actual.issubset(keys | optional_keys):
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys - optional_keys)}."
        )
    return result


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name=name)


def build_visualization_request(config: DictConfig) -> DatasetVisualizationRequest:
    """Resolve the strict Hydra mapping into the public request contract."""
    top = _exact(config, name="config", keys={"roots", "visualization"})
    roots = RuntimePathRoots.from_mapping(
        _exact(
            top["roots"],
            name="roots",
            keys={
                "project_root",
                "data_root",
                "checkpoint_root",
                "artifact_root",
                "output_root",
                "cache_root",
                "external_asset_root",
            },
        ),
        repository_root=PROJECT_ROOT,
    )
    resolver = PathResolver(roots)
    raw = _exact(
        top["visualization"],
        name="visualization",
        keys={
            "domain",
            "dataset_root",
            "output_video",
            "trajectory_id",
            "logical_scene_id",
            "camera_id",
            "fps",
            "crf",
            "history_frames",
        },
        optional_keys=frozenset({"court_overlay"}),
    )
    domain_text = _required_text(raw["domain"], name="visualization.domain")
    try:
        domain = DatasetVisualizationDomain(domain_text)
    except ValueError as error:
        raise ValueError(
            "visualization.domain must be court, blcs, or plcs."
        ) from error
    dataset_root = resolver.resolve(
        PathRole.DATA,
        _required_text(raw["dataset_root"], name="visualization.dataset_root"),
    )
    output_video = resolver.resolve(
        PathRole.OUTPUT,
        _required_text(raw["output_video"], name="visualization.output_video"),
    )
    fps_value = raw["fps"]
    if isinstance(fps_value, bool) or not isinstance(fps_value, (int, float)):
        raise TypeError("visualization.fps must be numeric.")
    crf = raw["crf"]
    history_frames = raw["history_frames"]
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise TypeError("visualization.crf must be an integer.")
    if isinstance(history_frames, bool) or not isinstance(history_frames, int):
        raise TypeError("visualization.history_frames must be an integer.")
    court_overlay = (
        _court_overlay_configuration(raw["court_overlay"])
        if "court_overlay" in raw
        else DEFAULT_COURT_OVERLAY_CONFIGURATION
    )
    return DatasetVisualizationRequest(
        domain=domain,
        dataset_root=dataset_root,
        output_video=output_video,
        trajectory_id=_optional_text(
            raw["trajectory_id"], name="visualization.trajectory_id"
        ),
        logical_scene_id=_optional_text(
            raw["logical_scene_id"], name="visualization.logical_scene_id"
        ),
        camera_id=_optional_text(raw["camera_id"], name="visualization.camera_id"),
        fps=float(fps_value),
        crf=crf,
        history_frames=history_frames,
        court_overlay=court_overlay,
    )


def _court_overlay_configuration(value: object) -> CourtOverlayConfiguration:
    raw = _exact(
        value,
        name="visualization.court_overlay",
        keys={
            "mode",
            "color_rgb",
            "background_color_rgb",
            "opacity",
            "depth_epsilon_m",
            "near_plane_m",
            "maximum_cells",
            "maximum_surface_faces",
            "maximum_projected_pixels",
        },
    )
    mode_text = _required_text(
        raw["mode"],
        name="visualization.court_overlay.mode",
    )
    try:
        mode = CourtOverlayMode(mode_text)
    except ValueError as error:
        raise ValueError(
            "visualization.court_overlay.mode must be semantic or "
            "trajectory_support_aabb."
        ) from error
    color_values = _color_values(raw["color_rgb"], name="color_rgb")
    background_values = _color_values(
        raw["background_color_rgb"],
        name="background_color_rgb",
    )
    integer_fields: dict[str, int] = {}
    for name in (
        "maximum_cells",
        "maximum_surface_faces",
        "maximum_projected_pixels",
    ):
        raw_value = raw[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(f"visualization.court_overlay.{name} must be an integer.")
        integer_fields[name] = raw_value
    return CourtOverlayConfiguration(
        mode=mode,
        color_rgb=(
            cast(int, color_values[0]),
            cast(int, color_values[1]),
            cast(int, color_values[2]),
        ),
        background_color_rgb=(
            cast(int, background_values[0]),
            cast(int, background_values[1]),
            cast(int, background_values[2]),
        ),
        opacity=_numeric(
            raw["opacity"],
            name="visualization.court_overlay.opacity",
        ),
        depth_epsilon_m=_numeric(
            raw["depth_epsilon_m"],
            name="visualization.court_overlay.depth_epsilon_m",
        ),
        near_plane_m=_numeric(
            raw["near_plane_m"],
            name="visualization.court_overlay.near_plane_m",
        ),
        maximum_cells=integer_fields["maximum_cells"],
        maximum_surface_faces=integer_fields["maximum_surface_faces"],
        maximum_projected_pixels=integer_fields["maximum_projected_pixels"],
    )


def _numeric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _color_values(value: object, *, name: str) -> tuple[object, object, object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"visualization.court_overlay.{name} must be a sequence.")
    values = tuple(value)
    if len(values) != 3:
        raise ValueError(
            f"visualization.court_overlay.{name} must contain three values."
        )
    return values[0], values[1], values[2]


def validate_dataset_visualization_boundary(config: DictConfig) -> None:
    """Validate all config fields without retaining a second runtime schema."""
    build_visualization_request(config)


register_boundary_validator(
    DATASET_VISUALIZATION_BOUNDARY,
    validate_dataset_visualization_boundary,
)


__all__ = [
    "DATASET_VISUALIZATION_BOUNDARY",
    "build_visualization_request",
    "validate_dataset_visualization_boundary",
]
