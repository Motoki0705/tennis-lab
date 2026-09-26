"""Strict typed runtime configuration for every tennis-scene entrypoint."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import (
    BUNDLED_MODEL_ASSET_SCHEMA,
    SUBMODULE_RUNTIME_SCHEMA,
    BundledModelAssetPaths,
    SubmoduleRuntimeConfig,
)
from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tasks.base.visualization import parse_view_3d
from src.tasks.base.visualization.orchestrator import parse_hw
from src.tasks.court_detection.inference.regions import CourtRegionSearchConfig
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionConfig
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryConfig
from src.tennis_scene.pipeline.components.court_kp import (
    CourtKPConfig,
    CourtKPPostprocessConfig,
)
from src.tennis_scene.pipeline.contracts import STANDARD_COMPONENTS
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    SemanticConfigurationError,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.rendering.camera_view import CameraController

_PATH_FIELDS = {
    f"{role.value}_root": ConfigField.of(str)
    for role in (
        PathRole.PROJECT,
        PathRole.DATA,
        PathRole.CHECKPOINT,
        PathRole.ARTIFACT,
        PathRole.OUTPUT,
        PathRole.CACHE,
        PathRole.EXTERNAL_ASSET,
    )
}
PATHS_SCHEMA = StrictConfigSchema(name="tennis_scene.paths", fields=_PATH_FIELDS)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _plain(cfg: DictConfig) -> Mapping[str, object]:
    value = OmegaConf.to_container(cfg, resolve=True)
    return _mapping(value, name="tennis_scene configuration")


def _roots(value: object) -> tuple[RuntimePathRoots, PathResolver]:
    mapping = _mapping(value, name="tennis_scene.paths")
    PATHS_SCHEMA.validate(mapping)
    roots = RuntimePathRoots.from_mapping(mapping, repository_root=PROJECT_ROOT)
    return roots, PathResolver(roots)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _nullable_sequence(value: object, *, name: str) -> tuple[object, ...] | None:
    if value is None:
        return None
    return _sequence(value, name=name)


def _nullable_string_sequence(value: object, *, name: str) -> tuple[str, ...] | None:
    values = _nullable_sequence(value, name=name)
    if values is None:
        return None
    if any(type(item) is not str for item in values):
        raise TypeError(f"{name} must contain exactly str values.")
    return tuple(cast(str, item) for item in values)


def _numeric_pair(value: object, *, name: str) -> tuple[float, float]:
    values = _sequence(value, name=name)
    if len(values) != 2:
        raise SemanticConfigurationError(f"{name} must contain exactly two values.")
    if any(type(item) not in (float, int) for item in values):
        raise TypeError(f"{name} must contain exactly two numeric values.")
    return float(cast(float | int, values[0])), float(cast(float | int, values[1]))


def _positive(value: int | float, *, name: str) -> None:
    if value <= 0:
        raise SemanticConfigurationError(f"{name} must be positive, got {value}.")


def _unit_interval(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise SemanticConfigurationError(f"{name} must be in [0, 1], got {value}.")


def _non_negative(value: float, *, name: str) -> None:
    if value < 0.0:
        raise SemanticConfigurationError(f"{name} must be non-negative, got {value}.")


def _window_contract(size: int, overlap: int, *, name: str) -> None:
    _positive(size, name=f"{name}.window_size")
    if overlap < 0 or overlap >= size:
        raise SemanticConfigurationError(
            f"{name}.window_overlap must satisfy 0 <= overlap < window_size; "
            f"got overlap={overlap}, window_size={size}."
        )


def _single_component(value: str, *, name: str) -> str:
    if not value or Path(value).name != value or value in {".", ".."}:
        raise SemanticConfigurationError(f"{name} must be one path component.")
    return value


_STAGE_IO_FIELDS = {
    "source": ConfigField.of(str),
    "save_result": ConfigField.of(bool),
    "output_path": ConfigField.of(str),
    "load_path": ConfigField.of(str, type(None)),
}


def _stage_path(
    stage: Mapping[str, object],
    resolver: PathResolver,
    *,
    name: str,
) -> tuple[Path | None, Path]:
    source = cast(str, stage["source"])
    if source not in {"execute", "load"}:
        raise SemanticConfigurationError(
            f"tennis_scene.{name}.source must be 'execute' or 'load'."
        )
    raw_load = stage["load_path"]
    if (source == "load") != (raw_load is not None):
        raise SemanticConfigurationError(
            f"tennis_scene.{name}: source='load' requires load_path, while "
            "source='execute' forbids it."
        )
    load_path = (
        resolver.resolve(PathRole.ARTIFACT, cast(str, raw_load))
        if raw_load is not None
        else None
    )
    output_path = resolver.resolve(PathRole.ARTIFACT, cast(str, stage["output_path"]))
    return load_path, output_path


_POSTPROCESS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.court_kp.postprocess",
    fields={
        field.name: ConfigField.of(
            type(getattr(CourtKPPostprocessConfig(), field.name))
        )
        for field in fields(CourtKPPostprocessConfig)
    },
)
_COURT_REGION_SCHEMA = StrictConfigSchema(
    name="tennis_scene.court_kp.region_search",
    fields={
        "enabled": ConfigField.of(bool),
        "min_inliers": ConfigField.of(int),
        "inlier_distance_ratio": ConfigField.of(float, int),
        "min_area_ratio": ConfigField.of(float, int),
    },
)
_COURT_SCHEMA = StrictConfigSchema(
    name="tennis_scene.court_kp",
    fields={"checkpoint": ConfigField.of(str), "subpixel_refine": ConfigField.of(bool),
            "postprocess": ConfigField.mapping(_POSTPROCESS_SCHEMA),
            "region_search": ConfigField.mapping(_COURT_REGION_SCHEMA)},
)
_PEOPLE_MODELS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.people_models",
    fields={
        **{name: ConfigField.of(str) for name in ("detector", "gvhmr_checkpoint", "yolo_checkpoint", "dino_checkpoint", "dino_repository", "vitpose_checkpoint", "hmr2_checkpoint", "body_models_dir")},
        "bundled_assets": ConfigField.mapping(BUNDLED_MODEL_ASSET_SCHEMA),
        "runtime": ConfigField.mapping(SUBMODULE_RUNTIME_SCHEMA),
    },
)
_TRAJECTORY_SCHEMA = StrictConfigSchema(
    name="tennis_scene.ball_detection.trajectory_gate",
    fields={
        "enabled": ConfigField.of(bool),
        "max_residual_px": ConfigField.of(float, int),
        "k_support": ConfigField.of(int),
        "max_support_gap": ConfigField.of(int),
        "max_passes": ConfigField.of(int),
    },
)
_BALL_SCHEMA = StrictConfigSchema(
    name="tennis_scene.ball_detection",
    fields={
        **_STAGE_IO_FIELDS,
        "enabled": ConfigField.of(bool),
        "checkpoint": ConfigField.of(str),
        "batch_size": ConfigField.of(int),
        "image_size": ConfigField.sequence(ConfigField.of(int)),
        "normalize_imagenet": ConfigField.of(bool),
        "score_threshold": ConfigField.of(float),
        "subpixel_refine": ConfigField.of(bool),
        "checkpoint_strict": ConfigField.of(bool),
        "checkpoint_weights_only": ConfigField.of(bool),
        "prefetch_batches": ConfigField.of(int),
        "window_stride": ConfigField.of(int, type(None)),
        "tail_policy": ConfigField.of(str),
        "overlap_aggregation": ConfigField.of(str),
        "pin_memory": ConfigField.of(bool),
        "trajectory_gate": ConfigField.mapping(_TRAJECTORY_SCHEMA),
    },
)
_FLAG_SCHEMA = StrictConfigSchema(name="tennis_scene.stage", fields={"enabled": ConfigField.of(bool)})
_PERSON_OBSERVATION_SCHEMA = StrictConfigSchema(name="tennis_scene.person_observations", fields={
    "enabled": ConfigField.of(bool), "visibility_threshold": ConfigField.of(float, int),
    "sideline_margin_m": ConfigField.of(float, int), "baseline_margin_m": ConfigField.of(float, int),
})
_AUTO_BALL_SCHEMA = StrictConfigSchema(name="tennis_scene.ball_detection", fields={k: v for k, v in _BALL_SCHEMA.fields.items() if k not in _STAGE_IO_FIELDS})
_FRAME_SAMPLING_SCHEMA = StrictConfigSchema(name="tennis_scene.frame_sampling", fields={"max_frames": ConfigField.of(int)})
_GEOMETRY_SCHEMA = StrictConfigSchema(name="tennis_scene.camera_geometry", fields={
    "reference_camera": ConfigField.of(str, type(None)), "calibration_samples": ConfigField.of(int),
    "consensus_ratio": ConfigField.of(float, int), "calibration_error_ratio": ConfigField.of(float, int),
    "side_min_frames": ConfigField.of(int), "side_max_cost": ConfigField.of(float, int),
    "side_min_support": ConfigField.of(float, int), "side_min_margin": ConfigField.of(float, int),
})
_PLACEMENT_SCHEMA = StrictConfigSchema(name="tennis_scene.player_reconstruction.placement", fields={
    name: ConfigField.of(int) if name in {"min_joints", "min_scale_pairs", "max_nfev"} else ConfigField.of(float, int)
    for name in TemporalPlacementConfig.__dataclass_fields__
})
_PLAYER_RECONSTRUCTION_SCHEMA = StrictConfigSchema(name="tennis_scene.player_reconstruction", fields={
    "enabled": ConfigField.of(bool), "reprojection_px": ConfigField.of(float, int), "joint_confidence": ConfigField.of(float, int),
    "placement": ConfigField.mapping(_PLACEMENT_SCHEMA),
})
_BALL_RECONSTRUCTION_SCHEMA = StrictConfigSchema(name="tennis_scene.ball_reconstruction", fields={
    "enabled": ConfigField.of(bool), "reprojection_px": ConfigField.of(float, int), "min_frames": ConfigField.of(int),
})
_CACHE_SCHEMA = StrictConfigSchema(name="tennis_scene.cache", fields={"directory": ConfigField.of(str), "source": ConfigField.of(str), "overwrite": ConfigField.of(bool)})
_EXECUTION_SCHEMA = StrictConfigSchema(name="tennis_scene.execution", fields={name: ConfigField.of(str) for name in STANDARD_COMPONENTS})
_PIPELINE_SCHEMA = StrictConfigSchema(name="tennis_scene.pipeline", fields={
    "execution": ConfigField.mapping(_EXECUTION_SCHEMA),
    "paths": ConfigField.mapping(PATHS_SCHEMA), "video_paths": ConfigField.sequence(ConfigField.of(str)),
    "camera_ids": ConfigField.sequence(ConfigField.of(str)), "output_name": ConfigField.of(str),
    "output_directory": ConfigField.of(str), "device": ConfigField.of(str), "max_frames": ConfigField.of(int, type(None)),
    "court_kp": ConfigField.mapping(_COURT_SCHEMA), "people_models": ConfigField.mapping(_PEOPLE_MODELS_SCHEMA),
    "person_observations": ConfigField.mapping(_PERSON_OBSERVATION_SCHEMA), "ball_detection": ConfigField.mapping(_AUTO_BALL_SCHEMA),
    "frame_sampling": ConfigField.mapping(_FRAME_SAMPLING_SCHEMA), "camera_geometry": ConfigField.mapping(_GEOMETRY_SCHEMA),
    "player_reconstruction": ConfigField.mapping(_PLAYER_RECONSTRUCTION_SCHEMA), "ball_reconstruction": ConfigField.mapping(_BALL_RECONSTRUCTION_SCHEMA),
    "gvhmr": ConfigField.mapping(_FLAG_SCHEMA), "cache": ConfigField.mapping(_CACHE_SCHEMA),
})


@dataclass(frozen=True, slots=True)
class PipelineRuntimeConfig:
    """Automatic stage configuration; dataset callers can leave run inputs unbound."""

    roots: RuntimePathRoots
    resolver: PathResolver
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    output_path: Path
    device: str
    max_frames: int | None
    court_kp: CourtKPConfig
    people: PeopleModelConfig
    ball_detection: BallDetectionConfig
    sampling_max_frames: int
    camera_geometry: CameraGeometryConfig
    human_vis_threshold: float
    person_roi_margins: tuple[float, float]
    player_reprojection_px: float
    joint_confidence: float
    player_placement: TemporalPlacementConfig
    ball_reprojection_px: float
    ball_min_frames: int
    cache_directory: Path
    cache_source: str
    cache_overwrite: bool
    enabled: Mapping[str, bool]
    processing_settings: Mapping[str, object]
    component_sources: Mapping[str, str]

    @classmethod
    def from_config(cls, cfg: DictConfig, *, bind_inputs: bool = True) -> PipelineRuntimeConfig:
        value = _PIPELINE_SCHEMA.validate(_plain(cfg))
        roots, resolver = _roots(value["paths"])
        raw_videos = _sequence(value["video_paths"], name="video_paths") if bind_inputs else ()
        raw_cameras = _sequence(value["camera_ids"], name="camera_ids") if bind_inputs else ()
        if bind_inputs and (not 3 <= len(raw_videos) <= 5 or len(raw_videos) != len(raw_cameras)):
            raise SemanticConfigurationError("Automatic reconstruction requires 3..5 synchronized video paths and camera IDs")
        camera_ids = tuple(cast(str, x) for x in raw_cameras)
        if len(set(camera_ids)) != len(camera_ids) or any(not x.strip() for x in camera_ids):
            raise SemanticConfigurationError("Camera IDs must be nonempty and unique")
        video_paths = tuple(resolver.resolve(PathRole.DATA, cast(str, x)) for x in raw_videos)
        output_name = _single_component(cast(str, value["output_name"]), name="output_name")
        output_path = resolver.resolve(PathRole.OUTPUT, cast(str, value["output_directory"]), f"{output_name}.npz")
        device = cast(str, value["device"])
        if not device.strip():
            raise SemanticConfigurationError("Pipeline device must be explicit")
        max_frames = cast(int | None, value["max_frames"])
        if max_frames is not None:
            _positive(max_frames, name="max_frames")
        cache = _mapping(value["cache"], name="cache")
        cache_source = cast(str, cache["source"])
        if cache_source not in ("execute", "load") or (cache_source == "load" and cache["overwrite"]):
            raise SemanticConfigurationError("Cache source must be execute/load; load forbids overwrite")
        cache_directory = resolver.resolve(PathRole.ARTIFACT, cast(str, cache["directory"]))
        court = _mapping(value["court_kp"], name="court_kp")
        court_config = CourtKPConfig(
            checkpoint=resolver.resolve(PathRole.CHECKPOINT, cast(str, court["checkpoint"])), source="execute", mode="model", device=device,
            subpixel_refine=cast(bool, court["subpixel_refine"]), num_keypoints=14, save_result=False,
            output_path=cache_directory / "court.component.json", load_path=None,
            postprocess=CourtKPPostprocessConfig(**dict(_mapping(court["postprocess"], name="court.postprocess"))),
            output_keypoint_contract="camera_view_v2", load_keypoint_contract=None, resolver=resolver,
            region_search=CourtRegionSearchConfig(**dict(_mapping(court["region_search"], name="court.region_search"))),
        )
        models = _mapping(value["people_models"], name="people_models")
        runtime = SubmoduleRuntimeConfig.from_mapping(_mapping(models["runtime"], name="people_models.runtime"))
        if runtime.device != device:
            raise SemanticConfigurationError("People model device must equal pipeline device")
        people = PeopleModelConfig(
            detector=cast(str, models["detector"]),
            dino_checkpoint=resolver.resolve(PathRole.CHECKPOINT, cast(str, models["dino_checkpoint"])),
            dino_repository=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["dino_repository"])),
            yolo_checkpoint=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["yolo_checkpoint"])),
            vitpose_checkpoint=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["vitpose_checkpoint"])),
            hmr2_checkpoint=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["hmr2_checkpoint"])),
            gvhmr_checkpoint=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["gvhmr_checkpoint"])),
            body_models_dir=resolver.resolve(PathRole.EXTERNAL_ASSET, cast(str, models["body_models_dir"])),
            bundled_assets=BundledModelAssetPaths.from_mapping(_mapping(models["bundled_assets"], name="bundled_assets"), resolver=resolver), runtime=runtime,
        )
        ball_settings = dict(_mapping(value["ball_detection"], name="ball_detection"))
        ball_config = build_ball_detection_config({**ball_settings, "source": "execute", "save_result": False, "load_path": None, "output_path": str(cache["directory"]) + "/ball.component.json"}, resolver, device=device)
        sampling_max_frames = cast(int, _mapping(value["frame_sampling"], name="frame_sampling")["max_frames"])
        _positive(sampling_max_frames, name="frame_sampling.max_frames")
        geometry = CameraGeometryConfig(**cast(dict[str, Any], dict(_mapping(value["camera_geometry"], name="camera_geometry"))))
        if bind_inputs and geometry.reference_camera is not None and geometry.reference_camera not in camera_ids:
            raise SemanticConfigurationError("Reference camera must be in the source camera IDs")
        person = _mapping(value["person_observations"], name="person_observations")
        player = _mapping(value["player_reconstruction"], name="player_reconstruction")
        ball = _mapping(value["ball_reconstruction"], name="ball_reconstruction")
        visibility = float(cast(float, person["visibility_threshold"]))
        _unit_interval(visibility, name="person visibility")
        margins = (float(cast(float, person["sideline_margin_m"])), float(cast(float, person["baseline_margin_m"])))
        for margin in margins:
            if not math.isfinite(margin) or margin < 0:
                raise SemanticConfigurationError("Court ROI margins must be finite and nonnegative")
        player_error, ball_error = float(cast(float, player["reprojection_px"])), float(cast(float, ball["reprojection_px"]))
        for error in (player_error, ball_error):
            if not math.isfinite(error) or error <= 0:
                raise SemanticConfigurationError("Reprojection thresholds must be finite and positive")
        joint_confidence = float(cast(float, player["joint_confidence"]))
        placement = TemporalPlacementConfig(**cast(dict[str, Any], dict(_mapping(player["placement"], name="player_reconstruction.placement"))))
        _unit_interval(joint_confidence, name="joint_confidence")
        _positive(cast(int, ball["min_frames"]), name="ball_min_frames")
        enabled = {key: cast(bool, _mapping(value[key], name=key)["enabled"]) for key in ("person_observations", "ball_detection", "player_reconstruction", "ball_reconstruction", "gvhmr")}
        enabled.update(court_kp=True, camera_geometry=True)
        from src.tennis_scene.pipeline.feature_flags import validate_requested_features
        validate_requested_features(enabled)
        component_sources = {name: str(mode) for name, mode in _mapping(value["execution"], name="execution").items()}
        if any(mode not in {"execute", "load"} for mode in component_sources.values()):
            raise SemanticConfigurationError("Component execution modes must be execute/load")
        settings = {key: item for key, item in value.items() if key not in {"paths", "video_paths", "camera_ids", "output_name", "output_directory", "cache", "max_frames"}}
        return cls(roots, resolver, video_paths, camera_ids, output_path, device, max_frames, court_config, people, ball_config,
            sampling_max_frames, geometry, visibility, margins, player_error, joint_confidence, placement, ball_error, cast(int, ball["min_frames"]),
            cache_directory, cache_source, cast(bool, cache["overwrite"]), enabled, settings, component_sources)


_EXPORT_SCHEMA = StrictConfigSchema(
    name="tennis_scene.export",
    fields={
        "fps": ConfigField.of(float, int, type(None)),
        "width": ConfigField.of(int, type(None)),
        "height": ConfigField.of(int, type(None)),
        "crf": ConfigField.of(int),
        "overwrite": ConfigField.of(bool),
    },
)


@dataclass(frozen=True, slots=True)
class ClipExportRuntimeConfig:
    """Resolved export configuration for one dataset video."""

    roots: RuntimePathRoots
    resolver: PathResolver
    source_directory: Path
    projects_path: Path
    dataset_id: str
    video_id: str
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    output_dir: Path
    fps: float | None
    width: int | None
    height: int | None
    crf: int
    overwrite: bool

    @classmethod
    def _from_validated(cls, value: Mapping[str, object]) -> ClipExportRuntimeConfig:
        roots, resolver = _roots(value["paths"])
        from src.tennis_scene.clip_studio.layout import discover_clip_studio_layout

        layout = discover_clip_studio_layout(
            resolver, cast(str, value["source_directory"])
        )
        export = _mapping(value["export"], name="export")
        width = cast(int | None, export["width"])
        height = cast(int | None, export["height"])
        if (width is None) != (height is None):
            raise SemanticConfigurationError(
                "export.width and export.height must be specified together."
            )
        fps_raw = cast(float | int | None, export["fps"])
        if fps_raw is not None:
            _positive(fps_raw, name="export.fps")
        if width is not None:
            _positive(width, name="export.width")
        if height is not None:
            _positive(height, name="export.height")
        crf = cast(int, export["crf"])
        if crf < 0 or crf > 51:
            raise SemanticConfigurationError("export.crf must be in [0, 51].")
        return cls(
            roots=roots,
            resolver=resolver,
            source_directory=layout.source_directory,
            projects_path=layout.projects_path,
            dataset_id=layout.dataset_id,
            video_id=layout.video_id,
            video_paths=layout.video_paths,
            camera_ids=layout.camera_ids,
            output_dir=layout.dataset_directory,
            fps=None if fps_raw is None else float(fps_raw),
            width=width,
            height=height,
            crf=crf,
            overwrite=cast(bool, export["overwrite"]),
        )


@dataclass(frozen=True, slots=True)
class ClipStudioGUIRuntimeConfig:
    """Validated GUI settings for one clip-studio process."""

    canvas_width: int
    tile_width: int
    cache_frames: int
    seek_grab_threshold: int
    window_name: str
    zoom_step: float
    port: int


@dataclass(frozen=True, slots=True)
class AudioSyncRuntimeConfig:
    """Validated audio synchronization settings."""

    sample_rate: int
    envelope_rate: float
    max_seconds: float | None


@dataclass(frozen=True, slots=True)
class ClipStudioRuntimeConfig:
    """Validated GUI and project creation boundary."""

    export: ClipExportRuntimeConfig
    dataset_id: str
    video_id: str
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    gui: ClipStudioGUIRuntimeConfig
    audio_sync: AudioSyncRuntimeConfig


_GUI_SCHEMA = StrictConfigSchema(
    name="tennis_scene.clip_studio.gui",
    fields={
        "canvas_width": ConfigField.of(int),
        "tile_width": ConfigField.of(int),
        "cache_frames": ConfigField.of(int),
        "seek_grab_threshold": ConfigField.of(int),
        "window_name": ConfigField.of(str),
        "zoom_step": ConfigField.of(float),
        "port": ConfigField.of(int),
    },
)
_AUDIO_SCHEMA = StrictConfigSchema(
    name="tennis_scene.clip_studio.audio_sync",
    fields={
        "sample_rate": ConfigField.of(int),
        "envelope_rate": ConfigField.of(float),
        "max_seconds": ConfigField.of(float, int, type(None)),
    },
)
_CLIP_STUDIO_SCHEMA = StrictConfigSchema(
    name="tennis_scene.clip_studio",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "source_directory": ConfigField.of(str),
        "gui": ConfigField.mapping(_GUI_SCHEMA),
        "audio_sync": ConfigField.mapping(_AUDIO_SCHEMA),
        "export": ConfigField.mapping(_EXPORT_SCHEMA),
    },
)


def parse_clip_studio_config(cfg: DictConfig) -> ClipStudioRuntimeConfig:
    """Discover and validate one canonical raw dataset video."""
    value = _CLIP_STUDIO_SCHEMA.validate(_plain(cfg))
    export = ClipExportRuntimeConfig._from_validated(value)
    gui = _mapping(value["gui"], name="gui")
    canvas_width = cast(int, gui["canvas_width"])
    tile_width = cast(int, gui["tile_width"])
    cache_frames = cast(int, gui["cache_frames"])
    seek_grab_threshold = cast(int, gui["seek_grab_threshold"])
    port = cast(int, gui["port"])
    if not 1 <= port <= 65535:
        raise SemanticConfigurationError("gui.port must be between 1 and 65535.")
    zoom_step = cast(float, gui["zoom_step"])
    for field_name, number in (
        ("canvas_width", canvas_width),
        ("tile_width", tile_width),
        ("cache_frames", cache_frames),
        ("seek_grab_threshold", seek_grab_threshold),
    ):
        _positive(number, name=f"gui.{field_name}")
    if zoom_step <= 1.0:
        raise SemanticConfigurationError("gui.zoom_step must be greater than 1.")
    window_name = cast(str, gui["window_name"])
    if not window_name:
        raise SemanticConfigurationError("gui.window_name must be non-empty.")
    audio = _mapping(value["audio_sync"], name="audio_sync")
    sample_rate = cast(int, audio["sample_rate"])
    envelope_rate = cast(float, audio["envelope_rate"])
    max_seconds_raw = cast(float | int | None, audio["max_seconds"])
    _positive(sample_rate, name="audio_sync.sample_rate")
    _positive(envelope_rate, name="audio_sync.envelope_rate")
    if max_seconds_raw is not None:
        _positive(max_seconds_raw, name="audio_sync.max_seconds")
    return ClipStudioRuntimeConfig(
        export=export,
        dataset_id=export.dataset_id,
        video_id=export.video_id,
        video_paths=export.video_paths,
        camera_ids=export.camera_ids,
        gui=ClipStudioGUIRuntimeConfig(
            canvas_width=canvas_width,
            tile_width=tile_width,
            cache_frames=cache_frames,
            seek_grab_threshold=seek_grab_threshold,
            window_name=window_name,
            zoom_step=zoom_step,
            port=port,
        ),
        audio_sync=AudioSyncRuntimeConfig(
            sample_rate=sample_rate,
            envelope_rate=envelope_rate,
            max_seconds=None if max_seconds_raw is None else float(max_seconds_raw),
        ),
    )


_STYLE_SCHEMA = StrictConfigSchema(
    name="tennis_scene.visualization.style",
    fields={
        "trail_length": ConfigField.of(int),
        "show_trail": ConfigField.of(bool),
        "figsize": ConfigField.sequence(ConfigField.of(int, float)),
        "player_representation": ConfigField.of(str),
        "mesh_alpha": ConfigField.of(float),
        "theme": ConfigField.of(str),
        "show_ball_shadow": ConfigField.of(bool),
        "show_player_shadow": ConfigField.of(bool),
        "show_player_trail": ConfigField.of(bool),
        "player_trail_length": ConfigField.of(int),
        "show_bounces": ConfigField.of(bool),
        "show_hud": ConfigField.of(bool),
        "show_minimap": ConfigField.of(bool),
    },
)
_CAMERA_SCHEMA = StrictConfigSchema(
    name="tennis_scene.visualization.camera",
    fields={
        "preset": ConfigField.of(str, type(None)),
        "elev": ConfigField.of(float, int, type(None)),
        "azim": ConfigField.of(float, int, type(None)),
        "zoom": ConfigField.of(float, int, type(None)),
        "mode": ConfigField.of(str),
        "orbit_period_s": ConfigField.of(float),
        "keyframes": ConfigField.of(list, tuple, type(None)),
    },
)
_VISUALIZATION_ASSETS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.visualization.assets",
    fields={
        "smpl_faces": ConfigField.of(str),
        "smpl_joint_regressor": ConfigField.of(str),
    },
)
_VISUALIZATION_SCHEMA = StrictConfigSchema(
    name="tennis_scene.visualization",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "assets": ConfigField.mapping(_VISUALIZATION_ASSETS_SCHEMA),
        "input": ConfigField.of(str),
        "output": ConfigField.of(str, type(None)),
        "preview_output": ConfigField.of(str),
        "display": ConfigField.of(bool),
        "start_frame": ConfigField.of(int),
        "end_frame": ConfigField.of(int, type(None)),
        "fps": ConfigField.of(float, int, type(None)),
        "dpi": ConfigField.of(int),
        "writer": ConfigField.of(str),
        "style": ConfigField.mapping(_STYLE_SCHEMA),
        "camera": ConfigField.mapping(_CAMERA_SCHEMA),
    },
)


@dataclass(frozen=True, slots=True)
class TennisSceneVisualizationStyleConfig:
    """Exact typed style settings specific to the integrated scene renderer."""

    trail_length: int
    show_trail: bool
    figsize: tuple[float, float]
    player_representation: Literal["smpl", "skeleton"]
    mesh_alpha: float
    theme: Literal["light", "dark"]
    show_ball_shadow: bool
    show_player_shadow: bool
    show_player_trail: bool
    player_trail_length: int
    show_bounces: bool
    show_hud: bool
    show_minimap: bool


@dataclass(frozen=True, slots=True)
class VisualizationRuntimeConfig:
    roots: RuntimePathRoots
    smpl_faces_path: Path
    smpl_joint_regressor_path: Path
    input_path: Path
    output_path: Path | None
    preview_output: Path
    display: bool
    start_frame: int
    end_frame: int | None
    fps: float | None
    dpi: int
    writer: str
    style: TennisSceneVisualizationStyleConfig
    camera: CameraController


def parse_visualization_config(cfg: DictConfig) -> VisualizationRuntimeConfig:
    """Validate visualization paths, style, and complete camera fields."""
    value = _VISUALIZATION_SCHEMA.validate(_plain(cfg))
    roots, resolver = _roots(value["paths"])
    assets = _mapping(value["assets"], name="assets")
    camera = parse_view_3d(_mapping(value["camera"], name="camera"))
    style = _mapping(value["style"], name="style")
    player_representation = cast(str, style["player_representation"])
    if player_representation not in {"smpl", "skeleton"}:
        raise SemanticConfigurationError(
            "style.player_representation must be 'smpl' or 'skeleton'."
        )
    theme = cast(str, style["theme"])
    if theme not in {"light", "dark"}:
        raise SemanticConfigurationError("style.theme must be 'light' or 'dark'.")
    trail_length = cast(int, style["trail_length"])
    player_trail_length = cast(int, style["player_trail_length"])
    _positive(trail_length, name="style.trail_length")
    _positive(player_trail_length, name="style.player_trail_length")
    mesh_alpha = cast(float, style["mesh_alpha"])
    _unit_interval(mesh_alpha, name="style.mesh_alpha")
    figsize = _numeric_pair(style["figsize"], name="style.figsize")
    _positive(figsize[0], name="style.figsize[0]")
    _positive(figsize[1], name="style.figsize[1]")
    raw_output = value["output"]
    fps_raw = cast(float | int | None, value["fps"])
    if fps_raw is not None:
        _positive(fps_raw, name="fps")
    start_frame = cast(int, value["start_frame"])
    end_frame = cast(int | None, value["end_frame"])
    if start_frame < 0:
        raise SemanticConfigurationError("start_frame must be >= 0.")
    if end_frame is not None and end_frame <= start_frame:
        raise SemanticConfigurationError("end_frame must be greater than start_frame.")
    dpi = cast(int, value["dpi"])
    _positive(dpi, name="dpi")
    writer = cast(str, value["writer"])
    if not writer:
        raise SemanticConfigurationError("writer must be non-empty.")
    return VisualizationRuntimeConfig(
        roots=roots,
        smpl_faces_path=resolver.resolve(
            PathRole.DATA, cast(str, assets["smpl_faces"])
        ),
        smpl_joint_regressor_path=resolver.resolve(
            PathRole.EXTERNAL_ASSET, cast(str, assets["smpl_joint_regressor"])
        ),
        input_path=resolver.resolve(PathRole.ARTIFACT, cast(str, value["input"])),
        output_path=None
        if raw_output is None
        else resolver.resolve(PathRole.OUTPUT, cast(str, raw_output)),
        preview_output=resolver.resolve(
            PathRole.OUTPUT, cast(str, value["preview_output"])
        ),
        display=cast(bool, value["display"]),
        start_frame=start_frame,
        end_frame=end_frame,
        fps=None if fps_raw is None else float(fps_raw),
        dpi=dpi,
        writer=writer,
        style=TennisSceneVisualizationStyleConfig(
            trail_length=trail_length,
            show_trail=cast(bool, style["show_trail"]),
            figsize=figsize,
            player_representation=cast(
                Literal["smpl", "skeleton"], player_representation
            ),
            mesh_alpha=mesh_alpha,
            theme=cast(Literal["light", "dark"], theme),
            show_ball_shadow=cast(bool, style["show_ball_shadow"]),
            show_player_shadow=cast(bool, style["show_player_shadow"]),
            show_player_trail=cast(bool, style["show_player_trail"]),
            player_trail_length=player_trail_length,
            show_bounces=cast(bool, style["show_bounces"]),
            show_hud=cast(bool, style["show_hud"]),
            show_minimap=cast(bool, style["show_minimap"]),
        ),
        camera=camera,
    )


_VISUALIZE_TASKS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.visualize_tasks",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "scene_path": ConfigField.of(str),
        "video_paths": ConfigField.sequence(ConfigField.of(str)),
        "tasks": ConfigField.sequence(ConfigField.of(str)),
        "start_frame": ConfigField.of(int),
        "end_frame": ConfigField.of(int, type(None)),
        "fps": ConfigField.of(float, int, type(None)),
        "kp_conf_threshold": ConfigField.of(float),
        "trail_length": ConfigField.of(int),
        "dpi": ConfigField.of(int),
        "output_directory": ConfigField.of(str),
    },
)
_VISUALIZATION_TASK_NAMES = frozenset(
    {"ball_detection", "court_kp", "gvhmr", "plcs", "blcs", "gvhmr_alignment", "person_observations", "player_reconstruction", "ball_reconstruction"}
)


@dataclass(frozen=True, slots=True)
class VisualizeTasksRuntimeConfig:
    scene_path: Path
    video_paths: tuple[Path, ...]
    tasks: tuple[str, ...]
    start_frame: int
    end_frame: int | None
    fps: float | None
    kp_conf_threshold: float
    trail_length: int
    dpi: int
    output_directory: Path


def parse_visualize_tasks_config(cfg: DictConfig) -> VisualizeTasksRuntimeConfig:
    """Validate per-task visualization without metadata or pickle fallback."""
    value = _VISUALIZE_TASKS_SCHEMA.validate(_plain(cfg))
    _, resolver = _roots(value["paths"])
    raw_fps = cast(float | int | None, value["fps"])
    if raw_fps is not None:
        _positive(raw_fps, name="fps")
    videos = tuple(
        resolver.resolve(PathRole.DATA, cast(str, item))
        for item in _sequence(value["video_paths"], name="video_paths")
    )
    if not videos:
        raise SemanticConfigurationError("video_paths must not be empty.")
    tasks = tuple(cast(str, item) for item in _sequence(value["tasks"], name="tasks"))
    if not tasks:
        raise SemanticConfigurationError("tasks must not be empty.")
    unknown_tasks = set(tasks) - _VISUALIZATION_TASK_NAMES
    if unknown_tasks:
        raise SemanticConfigurationError(
            f"Unknown visualization tasks: {sorted(unknown_tasks)}."
        )
    if len(set(tasks)) != len(tasks):
        raise SemanticConfigurationError("tasks must be unique.")
    start_frame = cast(int, value["start_frame"])
    end_frame = cast(int | None, value["end_frame"])
    if start_frame < 0:
        raise SemanticConfigurationError("start_frame must be >= 0.")
    if end_frame is not None and end_frame <= start_frame:
        raise SemanticConfigurationError("end_frame must be greater than start_frame.")
    kp_conf_threshold = cast(float, value["kp_conf_threshold"])
    _unit_interval(kp_conf_threshold, name="kp_conf_threshold")
    trail_length = cast(int, value["trail_length"])
    dpi = cast(int, value["dpi"])
    _positive(trail_length, name="trail_length")
    _positive(dpi, name="dpi")
    return VisualizeTasksRuntimeConfig(
        scene_path=resolver.resolve(PathRole.ARTIFACT, cast(str, value["scene_path"])),
        video_paths=videos,
        tasks=tasks,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=None if raw_fps is None else float(raw_fps),
        kp_conf_threshold=kp_conf_threshold,
        trail_length=trail_length,
        dpi=dpi,
        output_directory=resolver.resolve(
            PathRole.OUTPUT, cast(str, value["output_directory"])
        ),
    )


_GENERATE_SCHEMA = StrictConfigSchema(
    name="tennis_scene.generate_dataset",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "dataset_directory": ConfigField.of(str),
        "clip_ids": ConfigField.of(list, tuple, type(None)),
        "overwrite": ConfigField.of(bool),
        "continue_on_error": ConfigField.of(bool),
        "pipeline_overrides": ConfigField.sequence(ConfigField.of(str)),
    },
)


@dataclass(frozen=True, slots=True)
class GenerateDatasetRuntimeConfig:
    roots: RuntimePathRoots
    dataset_directory: Path
    clip_ids: tuple[str, ...] | None
    overwrite: bool
    continue_on_error: bool
    pipeline_overrides: tuple[str, ...]


def parse_generate_dataset_config(cfg: DictConfig) -> GenerateDatasetRuntimeConfig:
    """Validate dataset generation before reading manifests or composing stages."""
    value = _GENERATE_SCHEMA.validate(_plain(cfg))
    roots, resolver = _roots(value["paths"])
    ids = _nullable_string_sequence(value["clip_ids"], name="clip_ids")
    parsed_ids = None
    if ids is not None:
        from src.tennis_scene.generate_dataset.manifest import split_clip_id

        parsed_ids = tuple("/".join(split_clip_id(clip_id)) for clip_id in ids)
        if len(set(parsed_ids)) != len(parsed_ids):
            raise SemanticConfigurationError("clip_ids must be unique.")
    overrides = tuple(
        cast(str, item)
        for item in _sequence(value["pipeline_overrides"], name="pipeline_overrides")
    )
    if any(not override.strip() or "=" not in override for override in overrides):
        raise SemanticConfigurationError(
            "pipeline_overrides must contain non-empty Hydra key=value overrides."
        )
    override_keys = tuple(override.split("=", maxsplit=1)[0] for override in overrides)
    forbidden_keys = tuple(
        key for key in override_keys if key == "paths" or key.startswith("paths.")
    )
    if forbidden_keys:
        raise SemanticConfigurationError(
            "pipeline_overrides may not replace the generation boundary's path "
            f"authority: {forbidden_keys}."
        )
    return GenerateDatasetRuntimeConfig(
        roots=roots,
        dataset_directory=resolver.resolve(
            PathRole.DATA, cast(str, value["dataset_directory"])
        ),
        clip_ids=parsed_ids,
        overwrite=cast(bool, value["overwrite"]),
        continue_on_error=cast(bool, value["continue_on_error"]),
        pipeline_overrides=overrides,
    )


def build_ball_detection_config(
    settings: Mapping[str, object], resolver: PathResolver, *, device: str
) -> BallDetectionConfig:
    """Compose the shared, strictly validated scene ball detector contract."""
    ball = _BALL_SCHEMA.validate(settings)
    gate = _mapping(ball["trajectory_gate"], name="ball_detection.trajectory_gate")
    ball_load, ball_output = _stage_path(ball, resolver, name="ball_detection")
    image_size = parse_hw(ball["image_size"], name="ball_detection.image_size")
    batch_size = cast(int, ball["batch_size"])
    _positive(batch_size, name="ball_detection.batch_size")
    prefetch_batches = cast(int, ball["prefetch_batches"])
    if prefetch_batches < 0:
        raise SemanticConfigurationError(
            "ball_detection.prefetch_batches must be >= 0."
        )
    score_threshold = cast(float, ball["score_threshold"])
    _unit_interval(score_threshold, name="ball_detection.score_threshold")
    window_stride = cast(int | None, ball["window_stride"])
    if window_stride is not None:
        _positive(window_stride, name="ball_detection.window_stride")
    tail_policy = cast(str, ball["tail_policy"])
    if tail_policy not in {"drop", "backfill"}:
        raise SemanticConfigurationError(
            "ball_detection.tail_policy must be 'drop' or 'backfill'."
        )
    overlap_aggregation = cast(str, ball["overlap_aggregation"])
    if overlap_aggregation not in {"last_window_wins", "max_score"}:
        raise SemanticConfigurationError(
            "ball_detection.overlap_aggregation must be 'last_window_wins' "
            "or 'max_score'."
        )
    gate_residual = float(cast(float | int, gate["max_residual_px"]))
    gate_support = cast(int, gate["k_support"])
    gate_gap = cast(int, gate["max_support_gap"])
    gate_passes = cast(int, gate["max_passes"])
    _positive(gate_residual, name="ball_detection.trajectory_gate.max_residual_px")
    _positive(gate_support, name="ball_detection.trajectory_gate.k_support")
    if gate_gap < 0:
        raise SemanticConfigurationError(
            "ball_detection.trajectory_gate.max_support_gap must be >= 0."
        )
    _positive(gate_passes, name="ball_detection.trajectory_gate.max_passes")
    return BallDetectionConfig(
        checkpoint=resolver.resolve(
            PathRole.CHECKPOINT, cast(str, ball["checkpoint"])
        ),
        source=cast(Literal["execute", "load"], ball["source"]),
        batch_size=batch_size,
        device=device,
        image_size=image_size,
        normalize_imagenet=cast(bool, ball["normalize_imagenet"]),
        score_threshold=score_threshold,
        subpixel_refine=cast(bool, ball["subpixel_refine"]),
        checkpoint_strict=cast(bool, ball["checkpoint_strict"]),
        checkpoint_weights_only=cast(bool, ball["checkpoint_weights_only"]),
        prefetch_batches=prefetch_batches,
        window_stride=window_stride,
        tail_policy=tail_policy,
        overlap_aggregation=overlap_aggregation,
        pin_memory=cast(bool, ball["pin_memory"]),
        trajectory_gate=TrajectoryGateConfig(
            enabled=cast(bool, gate["enabled"]),
            max_residual_px=gate_residual,
            k_support=gate_support,
            max_support_gap=gate_gap,
            max_passes=gate_passes,
        ),
        save_result=cast(bool, ball["save_result"]),
        output_path=ball_output,
        load_path=ball_load,
        resolver=resolver,
    )


def validate_pipeline_boundary(cfg: DictConfig) -> None:
    """Validate the complete reconstruction pipeline boundary."""
    PipelineRuntimeConfig.from_config(cfg)


def validate_clip_studio_boundary(cfg: DictConfig) -> None:
    """Validate the complete clip-studio GUI boundary."""
    parse_clip_studio_config(cfg)


def validate_generate_dataset_boundary(cfg: DictConfig) -> None:
    """Validate the complete pseudo-annotation generation boundary."""
    parse_generate_dataset_config(cfg)


def validate_visualization_boundary(cfg: DictConfig) -> None:
    """Validate the complete integrated-scene visualization boundary."""
    parse_visualization_config(cfg)


def validate_visualize_tasks_boundary(cfg: DictConfig) -> None:
    """Validate the complete per-task visualization boundary."""
    parse_visualize_tasks_config(cfg)
