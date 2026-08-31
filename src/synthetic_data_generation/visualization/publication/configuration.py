"""Strict Hydra boundary for one complete publication request."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.visualization.publication.contracts import (
    PublicationArtifactName,
    PublicationDrawingSettings,
    PublicationRequest,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

PUBLICATION_BOUNDARY = "synthetic.publication_visualization"


def build_publication_request(config: DictConfig) -> PublicationRequest:
    """Resolve one exact Hydra composition into the typed publication request."""
    top = _exact(config, name="config", keys={"roots", "publication"})
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
        top["publication"],
        name="publication",
        keys={
            "scene_id",
            "scene_root",
            "output_bundle",
            "artifacts",
            "court",
            "blcs",
            "plcs",
            "captured",
            "drawing",
        },
    )
    court = _exact(
        raw["court"],
        name="publication.court",
        keys={"trajectory_id", "frame_indices"},
    )
    blcs = _exact(
        raw["blcs"],
        name="publication.blcs",
        keys={"logical_scene_id", "camera_id", "frame_indices", "camera_ids"},
    )
    plcs = _exact(
        raw["plcs"],
        name="publication.plcs",
        keys={"logical_scene_id", "camera_id", "frame_indices", "camera_ids"},
    )
    captured = _exact(
        raw["captured"],
        name="publication.captured",
        keys={"camera_ids"},
    )
    drawing = _exact(
        raw["drawing"],
        name="publication.drawing",
        keys={
            "dataset_size",
            "alignment_size",
            "figure_size",
            "overview_size",
            "gif_duration_ms",
            "frustum_depth_metres",
            "line_width",
            "font_size",
            "history_frames",
            "maximum_rendered_captured_cameras",
            "coincident_centre_tolerance_metres",
            "coincident_forward_angle_tolerance_degrees",
            "maximum_artifact_bytes",
            "maximum_bundle_bytes",
        },
    )
    return PublicationRequest(
        scene_id=_text(raw["scene_id"], name="publication.scene_id"),
        scene_root=resolver.resolve(
            PathRole.DATA,
            _text(raw["scene_root"], name="publication.scene_root"),
        ),
        output_bundle=resolver.resolve(
            PathRole.OUTPUT,
            _text(raw["output_bundle"], name="publication.output_bundle"),
        ),
        artifact_names=tuple(
            PublicationArtifactName(_text(value, name="publication.artifacts"))
            for value in _sequence(raw["artifacts"], name="publication.artifacts")
        ),
        court_trajectory_id=_text(
            court["trajectory_id"], name="publication.court.trajectory_id"
        ),
        court_frame_indices=_integer_tuple(
            court["frame_indices"], name="publication.court.frame_indices"
        ),
        blcs_logical_scene_id=_text(
            blcs["logical_scene_id"], name="publication.blcs.logical_scene_id"
        ),
        blcs_camera_id=_text(blcs["camera_id"], name="publication.blcs.camera_id"),
        blcs_frame_indices=_integer_tuple(
            blcs["frame_indices"], name="publication.blcs.frame_indices"
        ),
        blcs_camera_ids=_text_tuple(
            blcs["camera_ids"], name="publication.blcs.camera_ids"
        ),
        plcs_logical_scene_id=_text(
            plcs["logical_scene_id"], name="publication.plcs.logical_scene_id"
        ),
        plcs_camera_id=_text(plcs["camera_id"], name="publication.plcs.camera_id"),
        plcs_frame_indices=_integer_tuple(
            plcs["frame_indices"], name="publication.plcs.frame_indices"
        ),
        plcs_camera_ids=_text_tuple(
            plcs["camera_ids"], name="publication.plcs.camera_ids"
        ),
        captured_camera_ids=_text_tuple(
            captured["camera_ids"], name="publication.captured.camera_ids"
        ),
        drawing=PublicationDrawingSettings(
            dataset_size=_size(drawing["dataset_size"], name="drawing.dataset_size"),
            alignment_size=_size(
                drawing["alignment_size"], name="drawing.alignment_size"
            ),
            figure_size=_size(drawing["figure_size"], name="drawing.figure_size"),
            overview_size=_size(drawing["overview_size"], name="drawing.overview_size"),
            gif_duration_ms=_integer(
                drawing["gif_duration_ms"], name="drawing.gif_duration_ms"
            ),
            frustum_depth_metres=_number(
                drawing["frustum_depth_metres"], name="drawing.frustum_depth_metres"
            ),
            line_width=_number(drawing["line_width"], name="drawing.line_width"),
            font_size=_integer(drawing["font_size"], name="drawing.font_size"),
            history_frames=_integer(
                drawing["history_frames"], name="drawing.history_frames"
            ),
            maximum_rendered_captured_cameras=_integer(
                drawing["maximum_rendered_captured_cameras"],
                name="drawing.maximum_rendered_captured_cameras",
            ),
            coincident_centre_tolerance_metres=_number(
                drawing["coincident_centre_tolerance_metres"],
                name="drawing.coincident_centre_tolerance_metres",
            ),
            coincident_forward_angle_tolerance_degrees=_number(
                drawing["coincident_forward_angle_tolerance_degrees"],
                name="drawing.coincident_forward_angle_tolerance_degrees",
            ),
            maximum_artifact_bytes=_integer(
                drawing["maximum_artifact_bytes"], name="drawing.maximum_artifact_bytes"
            ),
            maximum_bundle_bytes=_integer(
                drawing["maximum_bundle_bytes"], name="drawing.maximum_bundle_bytes"
            ),
        ),
    )


def validate_publication_boundary(config: DictConfig) -> None:
    """Validate every configured field before publication side effects."""
    build_publication_request(config)


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


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _integer_tuple(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(_integer(item, name=name) for item in _sequence(value, name=name))


def _text_tuple(value: object, *, name: str) -> tuple[str, ...]:
    return tuple(_text(item, name=name) for item in _sequence(value, name=name))


def _size(value: object, *, name: str) -> tuple[int, int]:
    items = _integer_tuple(value, name=name)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly width and height.")
    return items[0], items[1]


register_boundary_validator(PUBLICATION_BOUNDARY, validate_publication_boundary)


__all__ = [
    "PUBLICATION_BOUNDARY",
    "build_publication_request",
    "validate_publication_boundary",
]
