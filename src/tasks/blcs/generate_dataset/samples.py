"""BLCS metrics, temporal adapters, and rendering for dataset sample GIFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.tasks.base.configuration import as_config_mapping, require_config_value
from src.tasks.base.generate_dataset.dataset_samples import (
    DatasetSampleCandidate,
    DatasetSamplesConfig,
    DatasetSampleSpec,
    RenderedDatasetSample,
    SelectedDatasetSample,
    assign_tercile,
    bounded_playback_fps,
    evenly_spaced_frame_indices,
    load_scene_visibility_summaries,
    materialize_dataset_samples,
    remap_sample_track_instances,
    save_animation_gif,
    select_stratified_samples,
    take_temporal_sample,
    tercile_boundaries,
    track_lifecycle_metrics,
    validate_sample_frame_indices,
)
from src.tasks.base.visualization.style import SceneStyleConfig
from src.tasks.blcs.generate_dataset.io.dataset_io import load_scene
from src.utils.hydra import register_boundary_validator
from src.utils.io import load_json

_SINGLE_PRIMARY_ORDER = ("deuce_side", "ad_side", "behind_baseline")
_MULTI_PRIMARY_ORDER = ("low", "medium", "high")
_DEUCE_CELLS = frozenset({0, 2, 4, 6})
_AD_CELLS = frozenset({1, 3, 5, 7})
_EVENT_FRAME_KEYS = (
    "t_start",
    "t_net",
    "t_bounce1",
    "t_bounce2",
    "t_bounce3",
    "t_return",
)
_SCENE_TEMPORAL_FIELDS = (
    "ball_pos_world",
    "ball_pos_norm",
    "ball_vel_world",
    "ball_vel_norm",
    "ball_present",
)
_CAMERA_TEMPORAL_FIELDS = ("ball_uv", "ball_vis")
_SAMPLE_STYLE = SceneStyleConfig(
    theme="dark",
    show_shadow=False,
    show_trail=False,
    trail_length=1,
    show_hud=False,
    show_minimap=False,
)


@dataclass(frozen=True, slots=True)
class _BLCSSceneStats:
    scene_id: str
    num_frames: int
    fps: float
    start_region: str
    start_cell: int
    start_cell_source: str
    rally_length: int
    end_reason: str
    winner_side: str | None
    initial_from_side: str
    track_count: int
    total_active_frames: int
    max_concurrent_tracks: int
    nested_shot_count: int
    camera_visibilities: tuple[float, ...]
    court_visibilities: tuple[float, ...]


def generate_blcs_dataset_samples(config: DatasetSamplesConfig) -> tuple[Path, ...]:
    """Generate stratified GIFs and manifests for every configured BLCS root."""
    manifests: list[Path] = []
    for spec in config.datasets:
        candidates, primary_order, strategy = _build_candidates(spec)
        selection = select_stratified_samples(
            candidates,
            primary_order=primary_order,
            samples_per_stratum=config.samples_per_stratum,
        )
        manifests.append(
            materialize_dataset_samples(
                task="blcs",
                spec=spec,
                config=config,
                selection=selection,
                strategy=strategy,
                render_sample=partial(_render_sample, spec, config),
            )
        )
    return tuple(manifests)


def _build_candidates(
    spec: DatasetSampleSpec,
) -> tuple[
    tuple[DatasetSampleCandidate, ...],
    tuple[str, str, str],
    Mapping[str, object],
]:
    stats = _load_scene_stats(spec)
    if spec.mode == "single":
        candidates = tuple(
            DatasetSampleCandidate(
                scene_id=scene.scene_id,
                primary_group=scene.start_region,
                duration_value=float(scene.num_frames),
                visibility_value=float(np.mean(scene.camera_visibilities)),
                auxiliary_value=float(scene.rally_length),
                camera_visibilities=scene.camera_visibilities,
                metrics=_metrics(scene),
            )
            for scene in stats
        )
        return (
            candidates,
            _SINGLE_PRIMARY_ORDER,
            {
                "primary_axis": (
                    "first non-serve from_cell region "
                    "(deuce/ad/behind-baseline; serve target fallback)"
                ),
                "duration_axis": "num_frames terciles within start region",
                "visibility_tie_break": "mean camera ball visibility Latin-square quantile",
                "auxiliary_tie_break": "rally_length",
            },
        )

    track_boundaries = tercile_boundaries(
        [float(scene.track_count) for scene in stats],
        metric_name="BLCS track_count",
    )
    candidates = tuple(
        DatasetSampleCandidate(
            scene_id=scene.scene_id,
            primary_group=assign_tercile(
                float(scene.track_count),
                track_boundaries,
                labels=_MULTI_PRIMARY_ORDER,
            ),
            duration_value=float(scene.total_active_frames),
            visibility_value=float(np.mean(scene.camera_visibilities)),
            auxiliary_value=float(scene.nested_shot_count),
            camera_visibilities=scene.camera_visibilities,
            metrics=_metrics(scene),
        )
        for scene in stats
    )
    return (
        candidates,
        _MULTI_PRIMARY_ORDER,
        {
            "primary_axis": "track_count global terciles",
            "primary_boundaries": list(track_boundaries),
            "duration_axis": "total_active_frames terciles within track-count band",
            "visibility_tie_break": "mean camera ball visibility Latin-square quantile",
            "auxiliary_tie_break": "nested shot count",
        },
    )


def _load_scene_stats(spec: DatasetSampleSpec) -> tuple[_BLCSSceneStats, ...]:
    scenes_dir = spec.root / "scenes"
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"BLCS scenes directory does not exist: {scenes_dir}")
    scene_dirs = tuple(sorted(path for path in scenes_dir.iterdir() if path.is_dir()))
    if not scene_dirs:
        raise ValueError(f"BLCS dataset contains no scene directories: {scenes_dir}")
    visibility = load_scene_visibility_summaries(
        spec.root,
        visibility_key="ball_visibility_ratio",
    )
    if set(visibility) != {path.name for path in scene_dirs}:
        raise ValueError(
            f"BLCS root metadata and scene directories differ for {spec.root}."
        )

    results: list[_BLCSSceneStats] = []
    for scene_dir in scene_dirs:
        meta_path = scene_dir / "meta.json"
        meta = as_config_mapping(load_json(meta_path), path=str(meta_path))
        scene_id = cast(
            "str", require_config_value(meta, "scene_id", str, path=str(meta_path))
        )
        if scene_id != scene_dir.name:
            raise ValueError(
                f"{meta_path}: scene_id {scene_id!r} does not match directory name."
            )
        num_frames = cast(
            "int", require_config_value(meta, "num_frames", int, path=str(meta_path))
        )
        fps_raw = require_config_value(
            meta, "fps_out", (float, int), path=str(meta_path)
        )
        fps = float(cast("float | int", fps_raw))
        if num_frames < 2 or not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{meta_path}: invalid BLCS frame count or fps_out.")
        rally_length = cast(
            "int", require_config_value(meta, "rally_length", int, path=str(meta_path))
        )
        if rally_length < 1:
            raise ValueError(f"{meta_path}: rally_length must be positive.")
        end_reason = cast(
            "str", require_config_value(meta, "end_reason", str, path=str(meta_path))
        )
        raw_winner = require_config_value(
            meta, "winner_side", (str, type(None)), path=str(meta_path)
        )
        winner_side = cast("str | None", raw_winner)
        initial_side = cast(
            "str",
            require_config_value(meta, "initial_from_side", str, path=str(meta_path)),
        )
        raw_shots = require_config_value(meta, "shots", list, path=str(meta_path))
        shots = _flatten_shots(
            cast("Sequence[object]", raw_shots), location=str(meta_path)
        )
        start_cell, start_source = resolve_start_cell(meta, shots)
        start_region = cell_region(start_cell)
        raw_tracks = require_config_value(
            meta, "track_instances", list, path=str(meta_path)
        )
        tracks = tuple(
            as_config_mapping(raw, path=f"{meta_path}.track_instances[{index}]")
            for index, raw in enumerate(cast("Sequence[object]", raw_tracks))
        )
        present_path = scene_dir / "ball_present.npy"
        if spec.mode == "single":
            if tracks or present_path.exists():
                raise ValueError(
                    f"{scene_dir}: configured single scene has tracking data."
                )
            track_count = 1
            total_active_frames = num_frames
            max_concurrent = 1
        else:
            if not tracks or not present_path.is_file():
                raise ValueError(
                    f"{scene_dir}: configured multi scene lacks tracking data."
                )
            track_count = len(tracks)
            total_active_frames, max_concurrent = track_lifecycle_metrics(
                tracks,
                num_frames=num_frames,
                location=str(meta_path),
            )
        camera_visibilities, court_visibilities = visibility[scene_id]
        results.append(
            _BLCSSceneStats(
                scene_id=scene_id,
                num_frames=num_frames,
                fps=fps,
                start_region=start_region,
                start_cell=start_cell,
                start_cell_source=start_source,
                rally_length=rally_length,
                end_reason=end_reason,
                winner_side=winner_side,
                initial_from_side=initial_side,
                track_count=track_count,
                total_active_frames=total_active_frames,
                max_concurrent_tracks=max_concurrent,
                nested_shot_count=len(shots),
                camera_visibilities=camera_visibilities,
                court_visibilities=court_visibilities,
            )
        )
    return tuple(results)


def _flatten_shots(
    raw_shots: Sequence[object],
    *,
    location: str,
) -> tuple[Mapping[str, object], ...]:
    shots: list[Mapping[str, object]] = []
    for index, raw in enumerate(raw_shots):
        path = f"{location}.shots[{index}]"
        record = as_config_mapping(raw, path=path)
        if "shots" in record:
            nested = require_config_value(record, "shots", list, path=path)
            shots.extend(
                as_config_mapping(shot, path=f"{path}.shots[{shot_index}]")
                for shot_index, shot in enumerate(cast("Sequence[object]", nested))
            )
        else:
            shots.append(record)
    return tuple(shots)


def resolve_start_cell(
    meta: Mapping[str, object],
    shots: Sequence[Mapping[str, object]],
) -> tuple[int, str]:
    """Resolve a physically meaningful first-hit cell for BLCS stratification."""
    for shot in shots:
        shot_type = require_config_value(shot, "shot_type", str, path="shot")
        if shot_type != "serve":
            cell = cast(
                "int", require_config_value(shot, "from_cell", int, path="shot")
            )
            cell_region(cell)
            return cell, "first_non_serve_from_cell"
    for shot in shots:
        cell = cast("int", require_config_value(shot, "to_cell", int, path="shot"))
        cell_region(cell)
        return cell, "serve_to_cell"
    cell = cast(
        "int", require_config_value(meta, "initial_from_cell", int, path="meta")
    )
    cell_region(cell)
    return cell, "initial_from_cell_no_shots"


def cell_region(cell: int) -> str:
    """Map canonical BLCS cell IDs to side-independent spatial strata."""
    if cell in _DEUCE_CELLS:
        return "deuce_side"
    if cell in _AD_CELLS:
        return "ad_side"
    if cell == 8:
        return "behind_baseline"
    raise ValueError(f"BLCS court cell must be within 0..8, got {cell}.")


def _metrics(scene: _BLCSSceneStats) -> Mapping[str, object]:
    return {
        "num_frames": scene.num_frames,
        "source_duration_seconds": scene.num_frames / scene.fps,
        "start_region": scene.start_region,
        "start_cell": scene.start_cell,
        "start_cell_source": scene.start_cell_source,
        "rally_length": scene.rally_length,
        "end_reason": scene.end_reason,
        "winner_side": scene.winner_side,
        "initial_from_side": scene.initial_from_side,
        "track_count": scene.track_count,
        "total_active_frames": scene.total_active_frames,
        "max_concurrent_tracks": scene.max_concurrent_tracks,
        "nested_shot_count": scene.nested_shot_count,
        "camera_visibility_mean": float(np.mean(scene.camera_visibilities)),
        "camera_visibility_min": min(scene.camera_visibilities),
        "camera_visibility_max": max(scene.camera_visibilities),
        "court_visibility_mean": float(np.mean(scene.court_visibilities)),
    }


def _render_sample(
    spec: DatasetSampleSpec,
    config: DatasetSamplesConfig,
    selected: SelectedDatasetSample,
    output_path: Path,
) -> RenderedDatasetSample:
    from src.tasks.blcs.visualization.rendering import BLCSSceneRenderer

    scene_path = spec.root / "scenes" / selected.candidate.scene_id
    scene = load_scene(
        scene_path,
        court_keypoint_contract=spec.court_keypoint_contract,
    )
    source_num_frames = len(scene["ball_pos_world"])
    meta_num_frames = scene["meta"]["num_frames"]
    if type(meta_num_frames) is not int or meta_num_frames != source_num_frames:
        raise ValueError(f"{scene_path}: BLCS meta/payload frame count mismatch.")
    source_fps = float(scene["meta"]["fps_out"])
    indices = evenly_spaced_frame_indices(source_num_frames, config.max_frames)
    encoded_fps = bounded_playback_fps(
        source_fps=source_fps,
        source_num_frames=source_num_frames,
        rendered_num_frames=len(indices),
        min_fps=config.min_fps,
        max_fps=config.max_fps,
    )
    sampled = subsample_blcs_scene(scene, indices=indices, playback_fps=encoded_fps)
    if selected.camera_index >= int(sampled["num_cameras"]):
        raise ValueError(
            f"{scene_path}: selected camera {selected.camera_index} is out of range."
        )
    renderer = BLCSSceneRenderer(style=_SAMPLE_STYLE)
    animation = renderer.create_animation(
        sampled,
        view=config.view,
        camera_idx=selected.camera_index,
        fps=float(encoded_fps),
        figsize=config.figure_size,
    )
    if animation is None:
        raise RuntimeError(f"BLCS renderer rejected sample view {config.view!r}.")
    save_animation_gif(
        animation,
        path=output_path,
        fps=encoded_fps,
        expected_frames=len(indices),
    )
    return RenderedDatasetSample(
        source_num_frames=source_num_frames,
        rendered_num_frames=len(indices),
        source_fps=source_fps,
        encoded_fps=encoded_fps,
        frame_indices=tuple(int(index) for index in indices),
    )


def subsample_blcs_scene(
    scene: dict[str, Any],
    *,
    indices: NDArray[np.int64],
    playback_fps: int,
) -> dict[str, Any]:
    """Return a non-mutating BLCS visualization proxy sampled on one timeline."""
    source_num_frames = len(scene["ball_pos_world"])
    validate_sample_frame_indices(indices, source_num_frames, task="BLCS")
    sampled = dict(scene)
    sampled["meta"] = deepcopy(scene["meta"])
    for field in _SCENE_TEMPORAL_FIELDS:
        if field in scene:
            sampled[field] = take_temporal_sample(
                scene[field],
                indices=indices,
                source_num_frames=source_num_frames,
                location=f"scene.{field}",
            )

    sampled_cameras: list[dict[str, Any]] = []
    for camera_index, raw_camera in enumerate(scene["cameras"]):
        camera = cast("dict[str, Any]", raw_camera)
        sampled_camera = dict(camera)
        for field in _CAMERA_TEMPORAL_FIELDS:
            sampled_camera[field] = take_temporal_sample(
                camera[field],
                indices=indices,
                source_num_frames=source_num_frames,
                location=f"scene.cameras[{camera_index}].{field}",
            )
        court_uv = np.asarray(camera["court_kp_uv"])
        court_vis = np.asarray(camera["court_kp_vis"])
        if (
            court_uv.ndim != 2
            or court_uv.shape[-1] != 2
            or court_vis.shape != court_uv.shape[:-1]
        ):
            raise ValueError(
                f"scene.cameras[{camera_index}] BLCS court arrays must be static (K,2)/(K,)."
            )
        ball_uv = np.asarray(sampled_camera["ball_uv"], dtype=np.float32).copy()
        ball_vis = np.asarray(sampled_camera["ball_vis"], dtype=np.bool_)
        if ball_uv.shape[:-1] != ball_vis.shape or ball_uv.shape[-1] != 2:
            raise ValueError(
                f"scene.cameras[{camera_index}] ball UV/visibility shapes disagree."
            )
        ball_uv[~ball_vis] = np.nan
        sampled_camera["ball_uv"] = ball_uv
        sampled_cameras.append(sampled_camera)
    sampled["cameras"] = sampled_cameras

    meta = cast("dict[str, Any]", sampled["meta"])
    meta["num_frames"] = len(indices)
    meta["fps_out"] = int(playback_fps)
    meta["shots"] = _remap_shot_records(
        meta["shots"],
        indices=indices,
        source_num_frames=source_num_frames,
    )
    if "ball_present" in sampled:
        remap_sample_track_instances(
            meta,
            np.asarray(sampled["ball_present"]),
            task="BLCS",
        )
    return sampled


def _remap_shot_records(
    raw_records: object,
    *,
    indices: NDArray[np.int64],
    source_num_frames: int,
) -> list[dict[str, object]]:
    if not isinstance(raw_records, list):
        raise ValueError("BLCS scene meta.shots must be a list.")
    remapped: list[dict[str, object]] = []
    for index, raw_record in enumerate(raw_records):
        record = dict(as_config_mapping(raw_record, path=f"shots[{index}]"))
        if "shots" in record:
            record["shots"] = _remap_shot_records(
                record["shots"],
                indices=indices,
                source_num_frames=source_num_frames,
            )
        else:
            for key in _EVENT_FRAME_KEYS:
                if key in record:
                    frame = record[key]
                    if type(frame) is not int:
                        raise ValueError(f"shots[{index}].{key} must be int.")
                    record[key] = _remap_frame(
                        frame,
                        indices=indices,
                        source_num_frames=source_num_frames,
                    )
        remapped.append(record)
    return remapped


def _remap_frame(
    frame: int,
    *,
    indices: NDArray[np.int64],
    source_num_frames: int,
) -> int:
    if frame < 0 or frame >= source_num_frames:
        return -1
    insertion = int(np.searchsorted(indices, frame))
    if insertion == 0:
        return 0
    if insertion == len(indices):
        return len(indices) - 1
    before = int(indices[insertion - 1])
    after = int(indices[insertion])
    return insertion - 1 if frame - before <= after - frame else insertion


def validate_dataset_samples_boundary(config: DictConfig) -> None:
    """Validate the BLCS dataset-sample Hydra boundary."""
    DatasetSamplesConfig.from_config(config, task="blcs")


register_boundary_validator(
    "blcs.generate_dataset_samples",
    validate_dataset_samples_boundary,
)


__all__ = [
    "cell_region",
    "generate_blcs_dataset_samples",
    "resolve_start_cell",
    "subsample_blcs_scene",
    "validate_dataset_samples_boundary",
]
