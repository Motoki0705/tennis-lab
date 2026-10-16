"""Strict Hydra boundary for the staged scene-reveal animation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.visualization.reveal.contracts import (
    RevealDurations,
    RevealMode,
    RevealNHTSettings,
    RevealOutput,
    RevealStyle,
    RevealViewpoint,
    SceneRevealRequest,
)
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT

SCENE_REVEAL_BOUNDARY = "synthetic.scene_reveal"

_ROOT_KEYS = {
    "project_root",
    "data_root",
    "checkpoint_root",
    "artifact_root",
    "output_root",
    "cache_root",
    "external_asset_root",
}


def _exact(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    mapping = cast(Mapping[str, object], value)
    if set(mapping) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(mapping))}, "
            f"unknown={sorted(set(mapping) - keys)}."
        )
    return mapping


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    return float(value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _numbers(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{name} must be a list of numbers.")
    return tuple(_number(item, name=name) for item in value)


def build_scene_reveal_request(config: DictConfig) -> SceneRevealRequest:
    """Resolve the strict Hydra mapping into the public request contract."""
    top = _exact(config, name="config", keys={"roots", "reveal"})
    resolver = PathResolver(
        RuntimePathRoots.from_mapping(
            _exact(top["roots"], name="roots", keys=_ROOT_KEYS),
            repository_root=PROJECT_ROOT,
        )
    )
    raw = _exact(
        top["reveal"],
        name="reveal",
        keys={
            "scene_root",
            "output_video",
            "work_directory",
            "mode",
            "still_times_seconds",
            "output",
            "durations_seconds",
            "viewpoint",
            "style",
            "nht",
        },
    )
    output = _exact(
        raw["output"], name="reveal.output", keys={"width", "height", "fps", "crf"}
    )
    durations = _exact(
        raw["durations_seconds"],
        name="reveal.durations_seconds",
        keys={
            "cameras",
            "sfm",
            "sfm_to_gaussians",
            "colorize",
            "splat",
            "view_dependence",
            "hold",
        },
    )
    viewpoint = _exact(
        raw["viewpoint"],
        name="reveal.viewpoint",
        keys={
            "anchor_court_id",
            "eye_court_m",
            "target_court_m",
            "up_court",
            "horizontal_fov_deg",
        },
    )
    style = _exact(
        raw["style"],
        name="reveal.style",
        keys={
            "background_rgb",
            "gaussian_point_rgb",
            "sfm_point_radius_px",
            "camera_rgb",
            "camera_frustum_depth_m",
            "camera_frustum_stride",
            "camera_line_width_px",
            "sfm_crop_court_m",
        },
    )
    nht = _exact(
        raw["nht"],
        name="reveal.nht",
        keys={"executable", "point_radius_m", "splat_stagger", "states_per_invocation"},
    )
    mode_text = _text(raw["mode"], name="reveal.mode")
    try:
        mode = RevealMode(mode_text)
    except ValueError as error:
        raise ValueError("reveal.mode must be video or stills.") from error
    return SceneRevealRequest(
        scene_root=resolver.resolve(
            PathRole.DATA, _text(raw["scene_root"], name="reveal.scene_root")
        ),
        output_video=resolver.resolve(
            PathRole.OUTPUT, _text(raw["output_video"], name="reveal.output_video")
        ),
        work_directory=resolver.resolve(
            PathRole.OUTPUT,
            _text(raw["work_directory"], name="reveal.work_directory"),
        ),
        mode=mode,
        still_times_seconds=_numbers(
            raw["still_times_seconds"], name="reveal.still_times_seconds"
        ),
        output=RevealOutput(
            width=_integer(output["width"], name="reveal.output.width"),
            height=_integer(output["height"], name="reveal.output.height"),
            fps=_number(output["fps"], name="reveal.output.fps"),
            crf=_integer(output["crf"], name="reveal.output.crf"),
        ),
        durations=RevealDurations.from_mapping(
            {
                key: _number(value, name=f"reveal.durations_seconds.{key}")
                for key, value in durations.items()
            }
        ),
        viewpoint=RevealViewpoint(
            anchor_court_id=_text(
                viewpoint["anchor_court_id"], name="reveal.viewpoint.anchor_court_id"
            ),
            eye_court_m=cast(
                tuple[float, float, float],
                _numbers(viewpoint["eye_court_m"], name="reveal.viewpoint.eye_court_m"),
            ),
            target_court_m=cast(
                tuple[float, float, float],
                _numbers(
                    viewpoint["target_court_m"], name="reveal.viewpoint.target_court_m"
                ),
            ),
            up_court=cast(
                tuple[float, float, float],
                _numbers(viewpoint["up_court"], name="reveal.viewpoint.up_court"),
            ),
            horizontal_fov_deg=_number(
                viewpoint["horizontal_fov_deg"],
                name="reveal.viewpoint.horizontal_fov_deg",
            ),
        ),
        style=RevealStyle(
            background_rgb=cast(
                tuple[float, float, float],
                _numbers(style["background_rgb"], name="reveal.style.background_rgb"),
            ),
            gaussian_point_rgb=cast(
                tuple[float, float, float],
                _numbers(
                    style["gaussian_point_rgb"], name="reveal.style.gaussian_point_rgb"
                ),
            ),
            sfm_point_radius_px=_integer(
                style["sfm_point_radius_px"], name="reveal.style.sfm_point_radius_px"
            ),
            camera_rgb=cast(
                tuple[float, float, float],
                _numbers(style["camera_rgb"], name="reveal.style.camera_rgb"),
            ),
            camera_frustum_depth_m=_number(
                style["camera_frustum_depth_m"],
                name="reveal.style.camera_frustum_depth_m",
            ),
            camera_frustum_stride=_integer(
                style["camera_frustum_stride"],
                name="reveal.style.camera_frustum_stride",
            ),
            camera_line_width_px=_integer(
                style["camera_line_width_px"], name="reveal.style.camera_line_width_px"
            ),
            sfm_crop_court_m=cast(
                tuple[float, float, float, float, float, float],
                _numbers(style["sfm_crop_court_m"], name="reveal.style.sfm_crop_court_m"),
            ),
        ),
        nht=RevealNHTSettings(
            executable=_text(nht["executable"], name="reveal.nht.executable"),
            point_radius_m=_number(
                nht["point_radius_m"], name="reveal.nht.point_radius_m"
            ),
            splat_stagger=_number(nht["splat_stagger"], name="reveal.nht.splat_stagger"),
            states_per_invocation=_integer(
                nht["states_per_invocation"], name="reveal.nht.states_per_invocation"
            ),
        ),
    )


def validate_scene_reveal_boundary(config: DictConfig) -> None:
    """Validate all config fields without retaining a second runtime schema."""
    build_scene_reveal_request(config)


register_boundary_validator(SCENE_REVEAL_BOUNDARY, validate_scene_reveal_boundary)


__all__ = [
    "SCENE_REVEAL_BOUNDARY",
    "build_scene_reveal_request",
    "validate_scene_reveal_boundary",
]
