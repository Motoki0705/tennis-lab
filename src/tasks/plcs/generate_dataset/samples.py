"""PLCS metrics, temporal adapters, and rendering for dataset sample GIFs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

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
from src.tasks.plcs.generate_dataset.io.scene_loader import AttrDict, load_scene
from src.utils.hydra import register_boundary_validator
from src.utils.io import load_json

_SINGLE_PRIMARY_ORDER = ("general", "walking", "running")
_MULTI_PRIMARY_ORDER = ("low", "medium", "high")
_SAMPLE_STYLE = SceneStyleConfig(
    theme="dark",
    show_shadow=False,
    show_trail=False,
    trail_length=1,
    show_hud=False,
    show_minimap=False,
)
_SCENE_TEMPORAL_FIELDS = (
    "position",
    "rotation",
    "canonical_pose_3d",
    "human_kp_3d",
    "person_present",
)
_CAMERA_TEMPORAL_FIELDS = (
    "human_kp_uv",
    "human_kp_vis",
    "court_kp_uv",
    "court_kp_vis",
)


@dataclass(frozen=True, slots=True)
class _PLCSSceneStats:
    scene_id: str
    num_frames: int
    fps: float
    motion_category: str
    gender: str
    track_count: int
    total_active_frames: int
    max_concurrent_tracks: int
    camera_visibilities: tuple[float, ...]
    court_visibilities: tuple[float, ...]


def generate_plcs_dataset_samples(config: DatasetSamplesConfig) -> tuple[Path, ...]:
    """Generate stratified GIFs and manifests for every configured PLCS root."""
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
                task="plcs",
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
                primary_group=scene.motion_category,
                duration_value=float(scene.num_frames),
                visibility_value=float(np.mean(scene.camera_visibilities)),
                auxiliary_value=0.0 if scene.gender == "male" else 1.0,
                camera_visibilities=scene.camera_visibilities,
                metrics=_metrics(scene),
            )
            for scene in stats
        )
        return (
            candidates,
            _SINGLE_PRIMARY_ORDER,
            {
                "primary_axis": "motion_category (general/walking/running)",
                "duration_axis": "num_frames terciles within motion_category",
                "visibility_tie_break": "mean camera human visibility Latin-square quantile",
                "auxiliary_tie_break": "gender diversity",
            },
        )

    track_boundaries = tercile_boundaries(
        [float(scene.track_count) for scene in stats],
        metric_name="PLCS track_count",
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
            auxiliary_value=float(scene.max_concurrent_tracks),
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
            "visibility_tie_break": "mean camera human visibility Latin-square quantile",
            "auxiliary_tie_break": "maximum concurrent tracks",
        },
    )


def _load_scene_stats(spec: DatasetSampleSpec) -> tuple[_PLCSSceneStats, ...]:
    scenes_dir = spec.root / "scenes"
    if not scenes_dir.is_dir():
        raise FileNotFoundError(f"PLCS scenes directory does not exist: {scenes_dir}")
    scene_dirs = tuple(sorted(path for path in scenes_dir.iterdir() if path.is_dir()))
    if not scene_dirs:
        raise ValueError(f"PLCS dataset contains no scene directories: {scenes_dir}")
    visibility = load_scene_visibility_summaries(
        spec.root,
        visibility_key="human_visibility_ratio",
    )
    if set(visibility) != {path.name for path in scene_dirs}:
        raise ValueError(
            f"PLCS root metadata and scene directories differ for {spec.root}."
        )

    results: list[_PLCSSceneStats] = []
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
        fps_raw = require_config_value(meta, "fps", (float, int), path=str(meta_path))
        fps = float(cast("float | int", fps_raw))
        if num_frames < 2 or not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{meta_path}: invalid PLCS frame count or fps.")
        category = cast(
            "str",
            require_config_value(meta, "motion_category", str, path=str(meta_path)),
        )
        if category not in _SINGLE_PRIMARY_ORDER:
            raise ValueError(f"{meta_path}: unknown motion_category {category!r}.")
        gender = cast(
            "str", require_config_value(meta, "gender", str, path=str(meta_path))
        )
        if gender not in {"female", "male"}:
            raise ValueError(f"{meta_path}: unsupported gender {gender!r}.")
        raw_tracks = require_config_value(
            meta, "track_instances", list, path=str(meta_path)
        )
        tracks = tuple(
            as_config_mapping(raw, path=f"{meta_path}.track_instances[{index}]")
            for index, raw in enumerate(cast("Sequence[object]", raw_tracks))
        )
        present_path = scene_dir / "person_present.npy"
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
            _PLCSSceneStats(
                scene_id=scene_id,
                num_frames=num_frames,
                fps=fps,
                motion_category=category,
                gender=gender,
                track_count=track_count,
                total_active_frames=total_active_frames,
                max_concurrent_tracks=max_concurrent,
                camera_visibilities=camera_visibilities,
                court_visibilities=court_visibilities,
            )
        )
    return tuple(results)


def _metrics(scene: _PLCSSceneStats) -> Mapping[str, object]:
    return {
        "num_frames": scene.num_frames,
        "source_duration_seconds": scene.num_frames / scene.fps,
        "motion_category": scene.motion_category,
        "gender": scene.gender,
        "track_count": scene.track_count,
        "total_active_frames": scene.total_active_frames,
        "max_concurrent_tracks": scene.max_concurrent_tracks,
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
    from src.tasks.plcs.visualization.rendering import PLCSSceneRenderer

    scene_path = spec.root / "scenes" / selected.candidate.scene_id
    scene = cast(
        "AttrDict",
        load_scene(
            scene_path,
            court_keypoint_contract=spec.court_keypoint_contract,
        ),
    )
    source_num_frames = len(scene.position)
    meta_num_frames = scene.meta["num_frames"]
    if type(meta_num_frames) is not int or meta_num_frames != source_num_frames:
        raise ValueError(f"{scene_path}: PLCS meta/payload frame count mismatch.")
    source_fps = float(scene.meta["fps"])
    indices = evenly_spaced_frame_indices(source_num_frames, config.max_frames)
    encoded_fps = bounded_playback_fps(
        source_fps=source_fps,
        source_num_frames=source_num_frames,
        rendered_num_frames=len(indices),
        min_fps=config.min_fps,
        max_fps=config.max_fps,
    )
    sampled = subsample_plcs_scene(scene, indices=indices, playback_fps=encoded_fps)
    if selected.camera_index >= int(sampled.num_cameras):
        raise ValueError(
            f"{scene_path}: selected camera {selected.camera_index} is out of range."
        )
    renderer = PLCSSceneRenderer(style=_SAMPLE_STYLE)
    animation = renderer.create_animation(
        sampled,
        view=config.view,
        camera_idx=selected.camera_index,
        fps=float(encoded_fps),
        figsize=config.figure_size,
    )
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


def subsample_plcs_scene(
    scene: AttrDict,
    *,
    indices: NDArray[np.int64],
    playback_fps: int,
) -> AttrDict:
    """Return a non-mutating PLCS visualization proxy sampled on one timeline."""
    source_num_frames = len(scene.position)
    validate_sample_frame_indices(indices, source_num_frames, task="PLCS")
    sampled = AttrDict(scene.copy())
    sampled.meta = deepcopy(scene.meta)
    for field in _SCENE_TEMPORAL_FIELDS:
        if field in scene:
            sampled[field] = take_temporal_sample(
                scene[field],
                indices=indices,
                source_num_frames=source_num_frames,
                location=f"scene.{field}",
            )

    sampled_cameras: list[AttrDict] = []
    for camera_index, camera in enumerate(scene.cameras):
        sampled_camera = AttrDict(camera.copy())
        for field in _CAMERA_TEMPORAL_FIELDS:
            sampled_camera[field] = take_temporal_sample(
                camera[field],
                indices=indices,
                source_num_frames=source_num_frames,
                location=f"scene.cameras[{camera_index}].{field}",
            )
        sampled_cameras.append(sampled_camera)
    sampled.cameras = sampled_cameras
    sampled.meta["num_frames"] = len(indices)
    sampled.meta["fps"] = float(playback_fps)
    if "person_present" in sampled:
        remap_sample_track_instances(
            sampled.meta,
            np.asarray(sampled.person_present),
            task="PLCS",
        )
    return sampled


def validate_dataset_samples_boundary(config: DictConfig) -> None:
    """Validate the PLCS dataset-sample Hydra boundary."""
    DatasetSamplesConfig.from_config(config, task="plcs")


register_boundary_validator(
    "plcs.generate_dataset_samples",
    validate_dataset_samples_boundary,
)


__all__ = [
    "generate_plcs_dataset_samples",
    "subsample_plcs_scene",
    "validate_dataset_samples_boundary",
]
