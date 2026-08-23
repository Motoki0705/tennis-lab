"""Strict typed runtime configuration for every tennis-scene entrypoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import (
    BUNDLED_MODEL_ASSET_SCHEMA,
    SUBMODULE_RUNTIME_SCHEMA,
    BundledModelAssetPaths,
    SubmoduleRuntimeConfig,
)
from src.tasks.ball_detection.inference.trajectory_gate import TrajectoryGateConfig
from src.tasks.base.configuration import CourtCoordinateNormalizationConfig
from src.tasks.base.visualization import parse_view_3d
from src.tasks.base.visualization.orchestrator import parse_hw
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionConfig
from src.tennis_scene.pipeline.components.blcs import BLCSConfig
from src.tennis_scene.pipeline.components.court_kp import (
    CourtKPConfig,
    CourtKPPostprocessConfig,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRConfig
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationConfig,
)
from src.tennis_scene.pipeline.components.plcs import PLCSConfig
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
from src.utils.schema.court_normalization import CourtCoordinateNormalization

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
    if value < 0.0 or value > 1.0:
        raise SemanticConfigurationError(f"{name} must be in [0, 1], got {value}.")


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
        "enabled": ConfigField.of(bool),
        "min_score": ConfigField.of(float),
        "ransac_reproj_threshold": ConfigField.of(float),
        "temporal_median_window": ConfigField.of(int),
    },
)
_COURT_SCHEMA = StrictConfigSchema(
    name="tennis_scene.court_kp",
    fields={
        **_STAGE_IO_FIELDS,
        "enabled": ConfigField.of(bool),
        "checkpoint": ConfigField.of(str),
        "frame_index": ConfigField.of(int),
        "mode": ConfigField.of(str),
        "num_keypoints": ConfigField.of(int),
        "subpixel_refine": ConfigField.of(bool),
        "postprocess": ConfigField.mapping(_POSTPROCESS_SCHEMA),
    },
)
_GVHMR_SCHEMA = StrictConfigSchema(
    name="tennis_scene.gvhmr",
    fields={
        **_STAGE_IO_FIELDS,
        "enabled": ConfigField.of(bool),
        "gvhmr_checkpoint": ConfigField.of(str),
        "detector": ConfigField.of(str),
        "yolo_checkpoint": ConfigField.of(str),
        "dino_checkpoint": ConfigField.of(str),
        "dino_repository": ConfigField.of(str),
        "vitpose_checkpoint": ConfigField.of(str),
        "hmr2_checkpoint": ConfigField.of(str),
        "body_models_dir": ConfigField.of(str),
        "bundled_assets": ConfigField.mapping(BUNDLED_MODEL_ASSET_SCHEMA),
        "runtime": ConfigField.mapping(SUBMODULE_RUNTIME_SCHEMA),
        "track_selection": ConfigField.of(str),
        "num_tracks": ConfigField.of(int),
    },
)
_ASSOCIATION_SCHEMA = StrictConfigSchema(
    name="tennis_scene.player_association",
    fields={
        **_STAGE_IO_FIELDS,
        "mode": ConfigField.of(str),
        "frame_index": ConfigField.of(int),
        "reference_camera": ConfigField.of(str, int),
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
_PLCS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.plcs",
    fields={
        **_STAGE_IO_FIELDS,
        "enabled": ConfigField.of(bool),
        "checkpoint": ConfigField.of(str),
        "window_size": ConfigField.of(int),
        "window_overlap": ConfigField.of(int),
        "human_vis_threshold": ConfigField.of(float),
    },
)
_BLCS_SCHEMA = StrictConfigSchema(
    name="tennis_scene.blcs",
    fields={
        **_STAGE_IO_FIELDS,
        "enabled": ConfigField.of(bool),
        "checkpoint": ConfigField.of(str),
        "window_size": ConfigField.of(int),
        "window_overlap": ConfigField.of(int),
    },
)
_PIPELINE_SCHEMA = StrictConfigSchema(
    name="tennis_scene.pipeline",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "court_coordinate_normalization": ConfigField.mapping(
            StrictConfigSchema(
                name="tennis_scene.court_coordinate_normalization",
                fields={"version": ConfigField.of(str)},
            )
        ),
        "video_paths": ConfigField.sequence(ConfigField.of(str)),
        "camera_ids": ConfigField.sequence(ConfigField.of(str)),
        "output_name": ConfigField.of(str),
        "device": ConfigField.of(str),
        "court_kp": ConfigField.mapping(_COURT_SCHEMA),
        "gvhmr": ConfigField.mapping(_GVHMR_SCHEMA),
        "player_association": ConfigField.mapping(_ASSOCIATION_SCHEMA),
        "ball_detection": ConfigField.mapping(_BALL_SCHEMA),
        "plcs": ConfigField.mapping(_PLCS_SCHEMA),
        "blcs": ConfigField.mapping(_BLCS_SCHEMA),
        "max_frames": ConfigField.of(int, type(None)),
    },
)


@dataclass(frozen=True, slots=True)
class PipelineRuntimeConfig:
    """Validated pipeline boundary and fully resolved stage paths."""

    roots: RuntimePathRoots
    resolver: PathResolver
    court_coordinate_normalization: CourtCoordinateNormalization
    video_paths: tuple[Path, ...]
    camera_ids: tuple[str, ...]
    output_path: Path
    device: str
    max_frames: int | None
    frame_index: int
    court_kp: CourtKPConfig
    gvhmr: GVHMRConfig
    player_association: PlayerAssociationConfig
    ball_detection: BallDetectionConfig
    plcs: PLCSConfig
    blcs: BLCSConfig
    enabled: Mapping[str, bool]

    @classmethod
    def from_config(cls, cfg: DictConfig) -> PipelineRuntimeConfig:
        """Reject the complete composed config before any model or I/O begins."""
        value = _PIPELINE_SCHEMA.validate(_plain(cfg))
        roots, resolver = _roots(value["paths"])
        court_coordinate_normalization = (
            CourtCoordinateNormalizationConfig.from_mapping(
                value["court_coordinate_normalization"]
            ).contract
        )
        raw_videos = _sequence(value["video_paths"], name="video_paths")
        raw_cameras = _sequence(value["camera_ids"], name="camera_ids")
        if not raw_videos or len(raw_videos) != len(raw_cameras):
            raise SemanticConfigurationError(
                "video_paths and camera_ids must be non-empty and have equal length."
            )
        video_paths = tuple(
            resolver.resolve(PathRole.DATA, cast(str, path)) for path in raw_videos
        )
        camera_ids = tuple(cast(str, item) for item in raw_cameras)
        if len(set(camera_ids)) != len(camera_ids):
            raise SemanticConfigurationError("camera_ids must be unique.")
        if any(not camera_id for camera_id in camera_ids):
            raise SemanticConfigurationError(
                "camera_ids must not contain empty values."
            )
        output_name = _single_component(
            cast(str, value["output_name"]), name="output_name"
        )
        output_path = resolver.resolve(
            PathRole.OUTPUT, "tennis_scene", f"{output_name}.npz"
        )
        device = cast(str, value["device"])
        if not device.strip():
            raise SemanticConfigurationError("device must be a non-empty string.")

        court = _mapping(value["court_kp"], name="court_kp")
        post = _mapping(court["postprocess"], name="court_kp.postprocess")
        court_load, court_output = _stage_path(court, resolver, name="court_kp")
        mode = cast(str, court["mode"])
        if mode not in {"model", "manual_ui"}:
            raise SemanticConfigurationError(
                "court_kp.mode must be model or manual_ui."
            )
        num_keypoints = cast(int, court["num_keypoints"])
        _positive(num_keypoints, name="court_kp.num_keypoints")
        frame_index = cast(int, court["frame_index"])
        if frame_index < 0:
            raise SemanticConfigurationError("court_kp.frame_index must be >= 0.")
        min_score = cast(float, post["min_score"])
        _unit_interval(min_score, name="court_kp.postprocess.min_score")
        ransac_threshold = cast(float, post["ransac_reproj_threshold"])
        _positive(
            ransac_threshold,
            name="court_kp.postprocess.ransac_reproj_threshold",
        )
        median_window = cast(int, post["temporal_median_window"])
        _positive(median_window, name="court_kp.postprocess.temporal_median_window")
        if median_window % 2 == 0:
            raise SemanticConfigurationError(
                "court_kp.postprocess.temporal_median_window must be odd."
            )
        court_config = CourtKPConfig(
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, court["checkpoint"])
            ),
            source=cast(Literal["execute", "load"], court["source"]),
            mode=cast(Literal["model", "manual_ui"], mode),
            device=device,
            subpixel_refine=cast(bool, court["subpixel_refine"]),
            num_keypoints=num_keypoints,
            save_result=cast(bool, court["save_result"]),
            output_path=court_output,
            load_path=court_load,
            postprocess=CourtKPPostprocessConfig(
                enabled=cast(bool, post["enabled"]),
                min_score=min_score,
                ransac_reproj_threshold=ransac_threshold,
                temporal_median_window=median_window,
            ),
            resolver=resolver,
        )

        gvhmr = _mapping(value["gvhmr"], name="gvhmr")
        gvhmr_load, gvhmr_output = _stage_path(gvhmr, resolver, name="gvhmr")
        gvhmr_runtime = SubmoduleRuntimeConfig.from_mapping(
            _mapping(gvhmr["runtime"], name="gvhmr.runtime")
        )
        bundled_assets = BundledModelAssetPaths.from_mapping(
            _mapping(gvhmr["bundled_assets"], name="gvhmr.bundled_assets"),
            resolver=resolver,
        )
        if gvhmr_runtime.device != device:
            raise SemanticConfigurationError(
                "gvhmr.runtime.device must equal the pipeline device."
            )
        num_tracks = cast(int, gvhmr["num_tracks"])
        _positive(num_tracks, name="gvhmr.num_tracks")
        gvhmr_config = GVHMRConfig(
            gvhmr_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["gvhmr_checkpoint"])
            ),
            source=cast(Literal["execute", "load"], gvhmr["source"]),
            detector=cast(str, gvhmr["detector"]),
            yolo_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["yolo_checkpoint"])
            ),
            dino_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["dino_checkpoint"])
            ),
            dino_repository=resolver.resolve(
                PathRole.EXTERNAL_ASSET, cast(str, gvhmr["dino_repository"])
            ),
            vitpose_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["vitpose_checkpoint"])
            ),
            hmr2_checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["hmr2_checkpoint"])
            ),
            body_models_dir=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, gvhmr["body_models_dir"])
            ),
            bundled_assets=bundled_assets,
            runtime=gvhmr_runtime,
            track_selection=cast(str, gvhmr["track_selection"]),
            num_tracks=num_tracks,
            save_result=cast(bool, gvhmr["save_result"]),
            output_path=gvhmr_output,
            load_path=gvhmr_load,
        )

        association = _mapping(value["player_association"], name="player_association")
        association_load, association_output = _stage_path(
            association, resolver, name="player_association"
        )
        association_mode = cast(str, association["mode"])
        if association_mode != "manual_ui":
            raise SemanticConfigurationError(
                "player_association.mode must be 'manual_ui'."
            )
        association_frame = cast(int, association["frame_index"])
        if association_frame < 0:
            raise SemanticConfigurationError(
                "player_association.frame_index must be >= 0."
            )
        reference_camera = cast(str | int, association["reference_camera"])
        if isinstance(reference_camera, int):
            if reference_camera < 0 or reference_camera >= len(camera_ids):
                raise SemanticConfigurationError(
                    "player_association.reference_camera index is out of range."
                )
        elif reference_camera not in camera_ids:
            raise SemanticConfigurationError(
                "player_association.reference_camera must name a configured camera."
            )
        association_config = PlayerAssociationConfig(
            source=cast(Literal["execute", "load"], association["source"]),
            mode=cast(Literal["manual_ui"], association_mode),
            initial_frame_index=association_frame,
            reference_camera=reference_camera,
            save_result=cast(bool, association["save_result"]),
            output_path=association_output,
            load_path=association_load,
        )

        ball = _mapping(value["ball_detection"], name="ball_detection")
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
        ball_config = BallDetectionConfig(
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

        plcs = _mapping(value["plcs"], name="plcs")
        plcs_load, plcs_output = _stage_path(plcs, resolver, name="plcs")
        plcs_window_size = cast(int, plcs["window_size"])
        plcs_window_overlap = cast(int, plcs["window_overlap"])
        _window_contract(plcs_window_size, plcs_window_overlap, name="plcs")
        human_vis_threshold = cast(float, plcs["human_vis_threshold"])
        _unit_interval(human_vis_threshold, name="plcs.human_vis_threshold")
        plcs_config = PLCSConfig(
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, plcs["checkpoint"])
            ),
            source=cast(Literal["execute", "load"], plcs["source"]),
            device=device,
            save_result=cast(bool, plcs["save_result"]),
            output_path=plcs_output,
            load_path=plcs_load,
            window_size=plcs_window_size,
            window_overlap=plcs_window_overlap,
            human_vis_threshold=human_vis_threshold,
            resolver=resolver,
            court_coordinate_normalization=court_coordinate_normalization,
        )
        blcs = _mapping(value["blcs"], name="blcs")
        blcs_load, blcs_output = _stage_path(blcs, resolver, name="blcs")
        blcs_window_size = cast(int, blcs["window_size"])
        blcs_window_overlap = cast(int, blcs["window_overlap"])
        _window_contract(blcs_window_size, blcs_window_overlap, name="blcs")
        blcs_config = BLCSConfig(
            checkpoint=resolver.resolve(
                PathRole.CHECKPOINT, cast(str, blcs["checkpoint"])
            ),
            source=cast(Literal["execute", "load"], blcs["source"]),
            device=device,
            save_result=cast(bool, blcs["save_result"]),
            output_path=blcs_output,
            load_path=blcs_load,
            window_size=blcs_window_size,
            window_overlap=blcs_window_overlap,
            resolver=resolver,
            court_coordinate_normalization=court_coordinate_normalization,
        )
        enabled = {
            name: cast(bool, section["enabled"])
            for name, section in (
                ("court_kp", court),
                ("gvhmr", gvhmr),
                ("ball_detection", ball),
                ("plcs", plcs),
                ("blcs", blcs),
            )
        }
        if not enabled["court_kp"] or not enabled["plcs"]:
            raise SemanticConfigurationError("court_kp and plcs must be enabled.")
        if enabled["plcs"] and not enabled["gvhmr"]:
            raise SemanticConfigurationError("plcs requires gvhmr.")
        if enabled["blcs"] and not enabled["ball_detection"]:
            raise SemanticConfigurationError("blcs requires ball_detection.")
        max_frames = cast(int | None, value["max_frames"])
        if max_frames is not None:
            _positive(max_frames, name="max_frames")
        return cls(
            roots=roots,
            resolver=resolver,
            court_coordinate_normalization=court_coordinate_normalization,
            video_paths=video_paths,
            camera_ids=camera_ids,
            output_path=output_path,
            device=device,
            max_frames=max_frames,
            frame_index=frame_index,
            court_kp=court_config,
            gvhmr=gvhmr_config,
            player_association=association_config,
            ball_detection=ball_config,
            plcs=plcs_config,
            blcs=blcs_config,
            enabled=enabled,
        )


_EXPORT_SCHEMA = StrictConfigSchema(
    name="tennis_scene.export",
    fields={
        "output_dir": ConfigField.of(str),
        "fps": ConfigField.of(float, int, type(None)),
        "width": ConfigField.of(int, type(None)),
        "height": ConfigField.of(int, type(None)),
        "crf": ConfigField.of(int),
        "overwrite": ConfigField.of(bool),
    },
)


@dataclass(frozen=True, slots=True)
class ClipExportRuntimeConfig:
    """Explicit project-file clip export configuration."""

    roots: RuntimePathRoots
    resolver: PathResolver
    project_path: Path
    clip_names: tuple[str, ...] | None
    output_dir: Path
    fps: float | None
    width: int | None
    height: int | None
    crf: int
    overwrite: bool

    @classmethod
    def _from_validated(cls, value: Mapping[str, object]) -> ClipExportRuntimeConfig:
        roots, resolver = _roots(value["paths"])
        export = _mapping(value["export"], name="export")
        names = _nullable_string_sequence(value["clip_names"], name="clip_names")
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
        parsed_names = None
        if names is not None:
            parsed_names = tuple(
                _single_component(item, name="clip_names item") for item in names
            )
            if len(set(parsed_names)) != len(parsed_names):
                raise SemanticConfigurationError("clip_names must be unique.")
        return cls(
            roots=roots,
            resolver=resolver,
            project_path=resolver.resolve(
                PathRole.ARTIFACT, cast(str, value["project_path"])
            ),
            clip_names=parsed_names,
            output_dir=resolver.resolve(
                PathRole.ARTIFACT, cast(str, export["output_dir"])
            ),
            fps=None if fps_raw is None else float(fps_raw),
            width=width,
            height=height,
            crf=crf,
            overwrite=cast(bool, export["overwrite"]),
        )


_EXPORT_BOUNDARY_SCHEMA = StrictConfigSchema(
    name="tennis_scene.export_clips",
    fields={
        "paths": ConfigField.mapping(PATHS_SCHEMA),
        "project_path": ConfigField.of(str),
        "clip_names": ConfigField.of(list, tuple, type(None)),
        "export": ConfigField.mapping(_EXPORT_SCHEMA),
    },
)


def parse_export_config(cfg: DictConfig) -> ClipExportRuntimeConfig:
    """Validate the headless export boundary."""
    value = _EXPORT_BOUNDARY_SCHEMA.validate(_plain(cfg))
    return ClipExportRuntimeConfig._from_validated(value)


@dataclass(frozen=True, slots=True)
class ClipStudioGUIRuntimeConfig:
    """Validated GUI settings for one clip-studio process."""

    canvas_width: int
    tile_width: int
    cache_frames: int
    seek_grab_threshold: int
    window_name: str
    zoom_step: float


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
    recording_id: str
    video_paths: tuple[Path, ...] | None
    camera_ids: tuple[str, ...] | None
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
        "project_path": ConfigField.of(str),
        "recording_id": ConfigField.of(str),
        "video_paths": ConfigField.of(list, tuple, type(None)),
        "camera_ids": ConfigField.of(list, tuple, type(None)),
        "gui": ConfigField.mapping(_GUI_SCHEMA),
        "audio_sync": ConfigField.mapping(_AUDIO_SCHEMA),
        "export": ConfigField.mapping(_EXPORT_SCHEMA),
    },
)


def parse_clip_studio_config(cfg: DictConfig) -> ClipStudioRuntimeConfig:
    """Validate one explicit project-file mode; no match-id aliases exist."""
    value = _CLIP_STUDIO_SCHEMA.validate(_plain(cfg))
    mutable = dict(value)
    mutable["clip_names"] = None
    export = ClipExportRuntimeConfig._from_validated(mutable)
    _, resolver = _roots(value["paths"])
    raw_videos = _nullable_string_sequence(value["video_paths"], name="video_paths")
    raw_cameras = _nullable_string_sequence(value["camera_ids"], name="camera_ids")
    if (raw_videos is None) != (raw_cameras is None):
        raise SemanticConfigurationError(
            "video_paths and camera_ids must be specified together."
        )
    if raw_videos is not None and (
        not raw_videos or len(raw_videos) != len(cast(tuple[str, ...], raw_cameras))
    ):
        raise SemanticConfigurationError(
            "video_paths and camera_ids must be non-empty and equal length."
        )
    camera_ids = None
    if raw_cameras is not None:
        camera_ids = raw_cameras
        if any(not camera_id for camera_id in camera_ids):
            raise SemanticConfigurationError(
                "camera_ids must not contain empty values."
            )
        if len(set(camera_ids)) != len(camera_ids):
            raise SemanticConfigurationError("camera_ids must be unique.")
    gui = _mapping(value["gui"], name="gui")
    canvas_width = cast(int, gui["canvas_width"])
    tile_width = cast(int, gui["tile_width"])
    cache_frames = cast(int, gui["cache_frames"])
    seek_grab_threshold = cast(int, gui["seek_grab_threshold"])
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
        recording_id=_single_component(
            cast(str, value["recording_id"]), name="recording_id"
        ),
        video_paths=None
        if raw_videos is None
        else tuple(resolver.resolve(PathRole.DATA, item) for item in raw_videos),
        camera_ids=camera_ids,
        gui=ClipStudioGUIRuntimeConfig(
            canvas_width=canvas_width,
            tile_width=tile_width,
            cache_frames=cache_frames,
            seek_grab_threshold=seek_grab_threshold,
            window_name=window_name,
            zoom_step=zoom_step,
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
    {"ball_detection", "court_kp", "gvhmr", "plcs", "blcs"}
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
        "court_coordinate_normalization": ConfigField.mapping(
            StrictConfigSchema(
                name="tennis_scene.generate_dataset.court_coordinate_normalization",
                fields={"version": ConfigField.of(str)},
            )
        ),
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
    court_coordinate_normalization: CourtCoordinateNormalization
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
    normalization_override_keys = tuple(
        key
        for key in override_keys
        if key == "court_coordinate_normalization"
        or key.startswith("court_coordinate_normalization.")
    )
    if normalization_override_keys:
        raise SemanticConfigurationError(
            "pipeline_overrides may not replace the generation boundary's "
            "court_coordinate_normalization selection; override the shared "
            f"Hydra group instead: {normalization_override_keys}."
        )
    return GenerateDatasetRuntimeConfig(
        roots=roots,
        court_coordinate_normalization=(
            CourtCoordinateNormalizationConfig.from_mapping(
                value["court_coordinate_normalization"]
            ).contract
        ),
        dataset_directory=resolver.resolve(
            PathRole.ARTIFACT, cast(str, value["dataset_directory"])
        ),
        clip_ids=parsed_ids,
        overwrite=cast(bool, value["overwrite"]),
        continue_on_error=cast(bool, value["continue_on_error"]),
        pipeline_overrides=overrides,
    )


def validate_pipeline_boundary(cfg: DictConfig) -> None:
    """Validate the complete reconstruction pipeline boundary."""
    PipelineRuntimeConfig.from_config(cfg)


def validate_clip_studio_boundary(cfg: DictConfig) -> None:
    """Validate the complete clip-studio GUI boundary."""
    parse_clip_studio_config(cfg)


def validate_export_clips_boundary(cfg: DictConfig) -> None:
    """Validate the complete headless clip-export boundary."""
    parse_export_config(cfg)


def validate_generate_dataset_boundary(cfg: DictConfig) -> None:
    """Validate the complete pseudo-annotation generation boundary."""
    parse_generate_dataset_config(cfg)


def validate_visualization_boundary(cfg: DictConfig) -> None:
    """Validate the complete integrated-scene visualization boundary."""
    parse_visualization_config(cfg)


def validate_visualize_tasks_boundary(cfg: DictConfig) -> None:
    """Validate the complete per-task visualization boundary."""
    parse_visualize_tasks_config(cfg)
