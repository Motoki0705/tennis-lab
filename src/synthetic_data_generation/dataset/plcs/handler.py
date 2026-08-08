"""Canonical CUDA/compact PLCS stage handler."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
    validate_court_transform_binding,
)
from src.synthetic_data_generation.composition.gaussians import (
    GaussianTensorSet,
    assign_instance_id,
    compose_foreground_frame_gaussians,
)
from src.synthetic_data_generation.configuration import PLCSDatasetConfiguration
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.assembler import (
    assemble_plcs_dataset,
    build_frame_label,
)
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    PreparedAvatar,
    prepare_avatar,
)
from src.synthetic_data_generation.dataset.plcs.diagnostics import (
    write_plcs_diagnostics,
)
from src.synthetic_data_generation.dataset.plcs.rendering.nht import NHTPLCSRenderer
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHDeviceModel,
    SMPLHModelData,
    load_smplh_model,
    upload_motion_clip,
    upload_smplh_model,
)
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSObjectTrack,
    build_global_timeline,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    ChunkWriter,
    DatasetPerformanceMetrics,
    ForegroundDeltaBatch,
    PerformanceTimer,
    RenderSession,
    directory_size_bytes,
)
from src.synthetic_data_generation.pipeline.contracts import (
    StageExecutionContext,
    StageExecutionSummary,
    StageName,
)
from src.synthetic_data_generation.pipeline.workspace import SceneWorkspace
from src.tasks.base.generate_dataset.camera_profiles import (
    CameraProfileConfig,
    assert_projection_equivalent,
    sample_camera_rig,
)
from src.tasks.base.generate_dataset.court_assignment import (
    CourtAssignment,
    assign_courts_balanced,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    ACCADMotionLibrary,
    MotionCategory,
    PLCSMotionClip,
    load_amass_motion_clip,
)


class PLCSAvatarAppearanceSource(Protocol):
    """Config-supplied explicit RGB appearance source for PLCS avatars."""

    def preflight(self, *, gaussian_count: int) -> None:
        """Validate RGB avatar appearance availability without output."""

    def load_avatar_appearance(
        self,
        *,
        clip: PLCSMotionClip,
        object_id: str,
        gaussian_count: int,
        seed: int,
        device: torch.device,
    ) -> AvatarAppearance:
        """Return explicit renderer-compatible features."""


@dataclass(frozen=True, slots=True)
class PLCSObjectRequest:
    """One explicit production avatar selection and court placement."""

    category: MotionCategory
    start_frame: int
    anchor_position_court_m: tuple[float, float, float]
    yaw_radians: float

    def __post_init__(self) -> None:
        if not isinstance(self.category, MotionCategory):
            raise TypeError("PLCS object category must be a MotionCategory.")
        if isinstance(self.start_frame, bool) or self.start_frame < 0:
            raise ValueError("PLCS object start_frame must be non-negative.")
        anchor = np.asarray(self.anchor_position_court_m, dtype=np.float64)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError(
                "PLCS object anchor must contain three finite court metres."
            )
        if not np.isfinite(self.yaw_radians):
            raise ValueError("PLCS object yaw must be finite.")


@dataclass(frozen=True, slots=True)
class PLCSStageParameters:
    """Explicit non-Hydra runtime inputs resolved by the composition root."""

    seed: int
    split: str
    scene_splits: Mapping[str, str]
    objects: tuple[PLCSObjectRequest, ...]
    smplh_model_root: Path
    gaussian_count: int
    smplh_batch_size: int
    device: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("PLCS seed must be a non-negative integer.")
        if not self.split.strip() or not self.scene_splits or not self.objects:
            raise ValueError("PLCS split, scene_splits, and objects must be explicit.")
        if not self.smplh_model_root.is_absolute():
            raise ValueError("SMPL-H model root must be an absolute resolved path.")
        if not self.smplh_model_root.is_dir():
            raise FileNotFoundError(
                f"SMPL-H model root does not exist: {self.smplh_model_root}"
            )
        for name, value in (
            ("gaussian_count", self.gaussian_count),
            ("smplh_batch_size", self.smplh_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise ValueError(
                "PLCS production parameters require an explicit CUDA device."
            )


@dataclass(slots=True)
class _PLCSStageCache:
    """Exact preflight/execute cache scoped to one handler attempt."""

    clips: dict[str, PLCSMotionClip] = field(default_factory=dict)
    models: dict[str, SMPLHModelData] = field(default_factory=dict)
    device_models: dict[str, SMPLHDeviceModel] = field(default_factory=dict)
    device_clips: dict[str, SMPLHDeviceClip] = field(default_factory=dict)
    validated_staging: Path | None = None

    def reset(self) -> None:
        self.clips.clear()
        self.models.clear()
        self.device_models.clear()
        self.device_clips.clear()
        self.validated_staging = None

    def clip(
        self,
        library: ACCADMotionLibrary,
        category: MotionCategory,
        *,
        seed: int,
    ) -> PLCSMotionClip:
        path = library.select_path(category, seed=seed)
        key = str(path)
        if key not in self.clips:
            self.clips[key] = load_amass_motion_clip(path, category=category)
        clip = self.clips[key]
        if clip.category is not category:
            raise ValueError("Cached PLCS clip category differs from its request.")
        return clip

    def prepare_device(
        self,
        clip: PLCSMotionClip,
        *,
        model_root: Path,
        device: torch.device,
    ) -> None:
        if clip.gender not in self.models:
            self.models[clip.gender] = load_smplh_model(
                model_root,
                gender=clip.gender,
            )
        model = self.models[clip.gender]
        if clip.gender not in self.device_models:
            self.device_models[clip.gender] = upload_smplh_model(model, device=device)
        if clip.source_path not in self.device_clips:
            self.device_clips[clip.source_path] = upload_motion_clip(
                clip,
                model,
                device=device,
            )


@dataclass(frozen=True, slots=True)
class PLCSStageHandler:
    """Generate one full-motion compact PLCS dataset on CUDA."""

    configuration: PLCSDatasetConfiguration
    camera_configuration: CameraProfileConfig
    motion_library: ACCADMotionLibrary
    avatar_appearance_source: PLCSAvatarAppearanceSource
    renderer: NHTPLCSRenderer
    parameters: PLCSStageParameters
    _cache: _PLCSStageCache = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=_PLCSStageCache,
    )

    def preflight(self, context: StageExecutionContext) -> None:
        """Load each selected clip/gender model once and retain device buffers."""
        paths = _stage_paths(context, require_staging=False)
        if not paths.scene_path.is_file() or paths.scene_path.is_symlink():
            raise FileNotFoundError(
                f"PLCS requires the canonical NHT scene export: {paths.scene_path}"
            )
        if (
            not self.configuration.require_articulated_motion
            or not self.configuration.multi_object_global_timeline
            or self.configuration.timeline.frame_selection != "all_source_frames"
        ):
            raise ValueError(
                "PLCS production configuration must require full articulated timelines."
            )
        device = torch.device(self.parameters.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PLCS production preflight requires available CUDA.")
        categories = set(self.configuration.motion_categories)
        requested_categories = {item.category.value for item in self.parameters.objects}
        if not requested_categories.issubset(categories):
            raise ValueError(
                "PLCS requested categories are not config-authorized: "
                f"{sorted(requested_categories - categories)}."
            )
        if (
            self.parameters.scene_splits.get(context.request.scene_id)
            != self.parameters.split
        ):
            raise ValueError(
                "PLCS scene split disagrees with balanced assignment input."
            )
        self._cache.reset()
        self.renderer.compositor.reset_stage()
        alignment = validate_alignment_outputs(paths.alignment_directory)
        assignments = assign_courts_balanced(
            self.parameters.scene_splits,
            layout=alignment.layout,
            seed=self.parameters.seed,
        )
        assignment = _current_assignment(
            assignments,
            scene_id=context.request.scene_id,
            split=self.parameters.split,
        )
        court = alignment.layout.court(assignment.court_instance_id)
        binding = TargetCourtBinding(
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            scene_from_court=court.scene_from_court,
            selection_seed=assignment.selection_seed,
        )
        tracks: list[PLCSObjectTrack] = []
        for index, request in enumerate(self.parameters.objects):
            clip = self._cache.clip(
                self.motion_library,
                request.category,
                seed=self.parameters.seed + index,
            )
            self._cache.prepare_device(
                clip,
                model_root=self.parameters.smplh_model_root,
                device=device,
            )
            tracks.append(_object_track(index=index, request=request, clip=clip))
        build_global_timeline(
            scene_id=context.request.scene_id,
            target_court=binding,
            tracks=tuple(tracks),
        )
        self.avatar_appearance_source.preflight(
            gaussian_count=self.parameters.gaussian_count,
        )

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Render all global frames into shared backgrounds and delta chunks."""
        timer = PerformanceTimer()
        paths = _stage_paths(context)
        if any(paths.staging_directory.iterdir()):
            raise ValueError(
                "PLCS stage requires an empty attempt-local staging directory."
            )
        device = torch.device(self.parameters.device)
        if not self._cache.clips or not self._cache.models:
            raise RuntimeError(
                "PLCS execute requires its successful stage-scoped preflight cache."
            )
        torch.cuda.reset_peak_memory_stats(device)
        alignment = validate_alignment_outputs(paths.alignment_directory)
        layout = alignment.layout
        assignments = assign_courts_balanced(
            self.parameters.scene_splits,
            layout=layout,
            seed=self.parameters.seed,
        )
        assignment = _current_assignment(
            assignments,
            scene_id=context.request.scene_id,
            split=self.parameters.split,
        )
        court = layout.court(assignment.court_instance_id)
        binding = TargetCourtBinding(
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            scene_from_court=court.scene_from_court,
            selection_seed=assignment.selection_seed,
        )
        validate_court_transform_binding(
            layout,
            court_instance_id=binding.court_instance_id,
            candidate_id=binding.candidate_id,
            transforms={
                "player_trajectory": binding.scene_from_court,
                "generated_camera": court.scene_from_court,
            },
        )
        rig = sample_camera_rig(
            self.camera_configuration,
            seed=self.parameters.seed,
            court=court,
        )
        projection_points = np.asarray(
            ((0.0, 0.0, 0.0), (-4.0, -10.0, 0.0), (4.0, 10.0, 1.0)),
            dtype=np.float64,
        )
        for sampled in rig.cameras:
            assert_projection_equivalent(
                sampled,
                court,
                projection_points,
                atol=1.0e-6,
            )
        resolutions = {
            (sampled.scene_camera.width, sampled.scene_camera.height)
            for sampled in rig.cameras
        }
        if len(resolutions) != 1:
            raise ValueError("PLCS compact chunks require one rig resolution.")
        width, height = next(iter(resolutions))

        tracks: list[PLCSObjectTrack] = []
        avatars: dict[str, PreparedAvatar] = {}
        for index, request in enumerate(self.parameters.objects):
            object_id = f"player-{index + 1:03d}"
            clip = self._cache.clip(
                self.motion_library,
                request.category,
                seed=self.parameters.seed + index,
            )
            if clip.gender not in self._cache.device_models:
                raise RuntimeError("PLCS preflight did not retain its device model.")
            appearance = self.avatar_appearance_source.load_avatar_appearance(
                clip=clip,
                object_id=object_id,
                gaussian_count=self.parameters.gaussian_count,
                seed=self.parameters.seed + index,
                device=device,
            )
            avatars[object_id] = prepare_avatar(
                asset_id=f"smplh-avatar-{index + 1:03d}",
                clip=clip,
                model=self._cache.models[clip.gender],
                device_model=self._cache.device_models[clip.gender],
                device_clip=self._cache.device_clips[clip.source_path],
                appearance=appearance,
                gaussian_count=self.parameters.gaussian_count,
                seed=self.parameters.seed + index,
            )
            tracks.append(_object_track(index=index, request=request, clip=clip))
        timeline = build_global_timeline(
            scene_id=context.request.scene_id,
            target_court=binding,
            tracks=tuple(tracks),
        )
        foreground_composition = timeline.to_foreground_composition(
            assets=tuple(
                avatars[track.object_id].semantic_asset for track in timeline.tracks
            ),
        )
        attempt_token = f"{context.request.scene_id}-plcs"
        session = RenderSession(
            domain="plcs",
            attempt_token=attempt_token,
            execution_device=str(device),
        )
        self.renderer.render_background_store(
            scene_path=paths.scene_path,
            cameras=tuple(sampled.scene_camera for sampled in rig.cameras),
            metric_adapter=alignment.metric_adapter,
            staging_directory=paths.staging_directory,
            session=session,
        )
        for sampled in rig.cameras:
            self.renderer.compositor.prepare_background(
                session.background("rig", sampled.scene_camera.camera_id),
                device=device,
            )
        if self.renderer.compositor.background_upload_count != len(rig.cameras):
            raise RuntimeError(
                "PLCS did not upload every static background exactly once."
            )

        writer = ChunkWriter(
            paths.staging_directory / "chunks",
            attempt_token=attempt_token,
            camera_ids=tuple(sampled.scene_camera.camera_id for sampled in rig.cameras),
            width=width,
            height=height,
        )
        chunk_readers: list[ChunkReader] = []
        chunk_size = self.configuration.timeline.chunk_size_frames
        for chunk_start in range(0, timeline.frame_count, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, timeline.frame_count)
            deltas = []
            labels = []
            for batch_start in range(
                chunk_start,
                chunk_stop,
                self.parameters.smplh_batch_size,
            ):
                batch_stop = min(
                    batch_start + self.parameters.smplh_batch_size,
                    chunk_stop,
                )
                frame_tensors: dict[str, dict[int, GaussianTensorSet]] = {}
                for track in timeline.tracks:
                    source_indices = tuple(
                        frame_index - track.start_frame
                        for frame_index in range(batch_start, batch_stop)
                        if track.start_frame <= frame_index < track.stop_frame
                    )
                    if source_indices:
                        frame_tensors[track.object_id] = avatars[
                            track.object_id
                        ].frame_tensors_batch(source_indices)
                for frame_index in range(batch_start, batch_stop):
                    frame = timeline.frames[frame_index]
                    expected_ids = tuple(
                        entry.instance_id for entry in frame.entries if entry.present
                    )
                    object_tensors = {
                        entry.object_id: assign_instance_id(
                            frame_tensors[entry.object_id][
                                int(entry.source_frame_index)
                            ],
                            entry.instance_id,
                        )
                        for entry in frame.entries
                        if entry.present and entry.source_frame_index is not None
                    }
                    composed = compose_foreground_frame_gaussians(
                        foreground_composition,
                        frame_index=frame_index,
                        object_tensors=object_tensors,
                    )
                    for camera_index, sampled in enumerate(rig.cameras):
                        delta, visibility = self.renderer.compositor.compose_delta(
                            frame_index=frame_index,
                            camera=sampled.scene_camera,
                            gaussians_scene=composed,
                            expected_instance_ids=expected_ids,
                        )
                        deltas.append(delta)
                        labels.append(
                            build_frame_label(
                                timeline=timeline,
                                rig=rig,
                                frame_index=frame_index,
                                camera_index=camera_index,
                                visibility=visibility,
                                seed=self.parameters.seed,
                            )
                        )
            chunk_readers.append(
                writer.write(
                    ForegroundDeltaBatch(
                        chunk_id=f"chunk-{chunk_start:06d}",
                        deltas=tuple(deltas),
                        metadata=tuple(labels),
                    )
                )
            )
        transient_generated_bytes = directory_size_bytes(paths.staging_directory)
        _discard_background_working_files(paths.staging_directory)
        diagnostic_paths = write_plcs_diagnostics(
            staging_directory=paths.staging_directory,
            timeline=timeline,
            rig=rig,
            avatars=avatars,
            assignments=assignments,
            court_instance_ids=tuple(
                court_instance.court_instance_id for court_instance in layout.courts
            ),
            clip_load_count=len(self._cache.clips),
            model_load_count=len(self._cache.models),
            execution_device=str(device),
        )
        all_diagnostics = (*diagnostic_paths, "diagnostics/performance.json")
        assembly = assemble_plcs_dataset(
            staging_directory=paths.staging_directory,
            timeline=timeline,
            rig=rig,
            chunk_readers=tuple(chunk_readers),
            attempt_token=attempt_token,
            chunk_size=chunk_size,
            diagnostics=all_diagnostics,
            seed=self.parameters.seed,
        )
        wall_seconds, cpu_seconds, peak_rss_bytes = timer.elapsed()
        sample_count = timeline.frame_count * len(rig.cameras)
        dense_reference_bytes = (
            sample_count
            * width
            * height
            * (
                3 * np.dtype(np.float32).itemsize
                + 2 * np.dtype(np.float32).itemsize
                + np.dtype(np.int32).itemsize
            )
        )
        metrics = DatasetPerformanceMetrics(
            domain="plcs",
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
            execution_device=str(device),
            cuda_peak_bytes=int(torch.cuda.max_memory_allocated(device)),
            nht_invocations=session.nht_invocations,
            background_cache_misses=session.background_cache_misses,
            complete_array_scans=sample_count,
            generated_bytes=transient_generated_bytes,
            published_bytes=0,
            dense_reference_bytes=dense_reference_bytes,
            frame_count=timeline.frame_count,
            camera_count=len(rig.cameras),
            sample_count=sample_count,
        )
        metrics = _write_performance_metrics(paths.staging_directory, metrics)
        metrics.validate_budget(self.configuration.performance)
        self._cache.validated_staging = paths.staging_directory
        return StageExecutionSummary(
            values={
                "mode": timeline.mode,
                "source_motion_count": len(timeline.tracks),
                "global_frame_count": timeline.frame_count,
                "planned_frame_count": timeline.frame_count,
                "rendered_frame_count": timeline.frame_count,
                "labelled_frame_count": timeline.frame_count,
                "camera_count": len(rig.cameras),
                "sample_count": assembly.sample_count,
                "chunk_count": assembly.chunk_count,
                "target_court_instance_id": binding.court_instance_id,
                "continuity_record_count": assembly.continuity.record_count,
                "clip_load_count": len(self._cache.clips),
                "model_load_count": len(self._cache.models),
                "nht_invocations": metrics.nht_invocations,
                "background_cache_misses": metrics.background_cache_misses,
                "execution_device": metrics.execution_device,
                "cuda_peak_bytes": metrics.cuda_peak_bytes,
                "published_bytes": metrics.published_bytes,
                "dense_reference_bytes": metrics.dense_reference_bytes,
            }
        )

    def validate(self, context: StageExecutionContext) -> None:
        """Accept only the exact staging tree already streamed by the assembler."""
        paths = _stage_paths(context)
        if self._cache.validated_staging != paths.staging_directory:
            raise RuntimeError(
                "PLCS staging did not complete its final streaming validation."
            )
        performance = paths.staging_directory / "diagnostics" / "performance.json"
        if not performance.is_file() or performance.is_symlink():
            raise FileNotFoundError(
                "PLCS performance diagnostics are missing after assembly."
            )
        metrics = DatasetPerformanceMetrics.from_dict(
            json.loads(performance.read_text(encoding="utf-8"))
        )
        metrics.validate_budget(self.configuration.performance)


@dataclass(frozen=True, slots=True)
class _PLCSStagePaths:
    scene_path: Path
    alignment_directory: Path
    staging_directory: Path


def _stage_paths(
    context: StageExecutionContext,
    *,
    require_staging: bool = True,
) -> _PLCSStagePaths:
    if context.stage.name is not StageName.PLCS_DATASET:
        raise ValueError("PLCSStageHandler received a non-PLCS stage context.")
    owner = Path(context.owner_path)
    if owner.parts[-2:] != ("datasets", "plcs"):
        raise ValueError("PLCS stage owner must be the fixed datasets/plcs directory.")
    root = owner.parents[1]
    workspace = SceneWorkspace(scene_id=context.request.scene_id, root=root)
    if workspace.owner_path(context.stage) != owner:
        raise ValueError(
            "PLCS context owner disagrees with the canonical SceneWorkspace."
        )
    expected_staging = workspace.staging_path(context.stage)
    if context.staging_path != expected_staging:
        raise ValueError(
            f"PLCS handler requires runner staging {expected_staging}, "
            f"got {context.staging_path}."
        )
    if require_staging and (
        not expected_staging.is_dir() or expected_staging.is_symlink()
    ):
        raise ValueError("PLCS staging must be an ordinary existing directory.")
    return _PLCSStagePaths(
        scene_path=root / "reconstruction" / "export" / "scene.json",
        alignment_directory=root / "alignment",
        staging_directory=expected_staging,
    )


def _current_assignment(
    assignments: tuple[CourtAssignment, ...],
    *,
    scene_id: str,
    split: str,
) -> CourtAssignment:
    matches = [
        assignment
        for assignment in assignments
        if assignment.scene_id == scene_id and assignment.split == split
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one balanced target-court assignment for {scene_id!r}/{split!r}."
        )
    return matches[0]


def _object_track(
    *,
    index: int,
    request: PLCSObjectRequest,
    clip: PLCSMotionClip,
) -> PLCSObjectTrack:
    return PLCSObjectTrack(
        object_id=f"player-{index + 1:03d}",
        instance_id=index + 1,
        asset_id=f"smplh-avatar-{index + 1:03d}",
        clip=clip,
        start_frame=request.start_frame,
        anchor_position_court_m=request.anchor_position_court_m,
        yaw_radians=request.yaw_radians,
    )


def _write_performance_metrics(
    staging: Path,
    metrics: DatasetPerformanceMetrics,
) -> DatasetPerformanceMetrics:
    path = staging / "diagnostics" / "performance.json"
    current = metrics
    for _ in range(8):
        path.write_text(
            json.dumps(current.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        size = directory_size_bytes(staging)
        if size == current.published_bytes:
            return current
        current = replace(
            current,
            generated_bytes=max(current.generated_bytes, size),
            published_bytes=size,
        )
    raise RuntimeError("PLCS published byte count did not reach a stable fixed point.")


def _discard_background_working_files(staging: Path) -> None:
    output = staging / "nht-backgrounds"
    request = staging / "nht-background-cameras.json"
    if output.is_symlink() or request.is_symlink():
        raise ValueError("NHT background working paths must not be symbolic links.")
    if output.is_dir():
        shutil.rmtree(output)
    elif output.exists():
        raise ValueError("NHT background output is not a directory.")
    if request.is_file():
        request.unlink()
    elif request.exists():
        raise ValueError("NHT background request is not an ordinary file.")


__all__ = [
    "PLCSAvatarAppearanceSource",
    "PLCSObjectRequest",
    "PLCSStageHandler",
    "PLCSStageParameters",
]
