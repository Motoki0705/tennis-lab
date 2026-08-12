"""Strict Hydra boundary for canonical dataset visualization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.visualization.contracts import (
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


def _exact(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    result = _mapping(value, name=name)
    if set(result) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(result))}, "
            f"unknown={sorted(set(result) - keys)}."
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
    )


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
