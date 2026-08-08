"""Canonical complete-scene CUDA/compact PLCS stage handler."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
    validate_court_transform_binding,
)
from src.synthetic_data_generation.configuration import PLCSDatasetConfiguration
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCSSceneAssemblyInput,
    PLCSSupervisionArrays,
    assemble_plcs_dataset,
    build_frame_label,
)
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    PLCSAvatarFrameTensors,
    compose_prevalidated_frame_gaussians,
)
from src.synthetic_data_generation.dataset.plcs.diagnostics import (
    write_plcs_diagnostics,
)
from src.synthetic_data_generation.dataset.plcs.execution import (
    CUDAPLCSExecutionBackend,
    PLCSExecutionBackend,
    PLCSPreparedAvatar,
)
from src.synthetic_data_generation.dataset.plcs.rendering.nht import NHTPLCSRenderer
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSGlobalTimeline,
    PLCSLogicalScene,
    PLCSObjectTrack,
    PLCSSceneInventory,
    build_global_timeline,
)
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    ChunkWriter,
    DatasetPerformanceMetrics,
    ForegroundDelta,
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
from src.synthetic_data_generation.scene_contract import MultiCourtLayout, SceneCamera
from src.tasks.base.generate_dataset.camera_profiles import (
    CameraProfileConfig,
    SampledCameraRig,
    assert_projection_equivalent,
    sample_camera_rig,
)
from src.tasks.base.generate_dataset.court_assignment import (
    CourtAssignment,
    assign_courts_balanced,
)
from src.tasks.plcs.data.targets import smplh_joints_to_coco17
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    ACCADMotionLibrary,
    MotionCategory,
    PLCSMotionClip,
    load_amass_motion_clip,
)
from src.utils.schema.court import (
    COURT_COORD_SCALE_XYZ,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
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
        if any(
            not scene_id.strip() or split not in {"train", "validation", "test"}
            for scene_id, split in self.scene_splits.items()
        ):
            raise ValueError("PLCS scene_splits contains an invalid scene or split.")
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
    """Host source cache scoped to one handler attempt."""

    clips: dict[str, PLCSMotionClip] = field(default_factory=dict)
    models: dict[str, object] = field(default_factory=dict)
    validated_staging: Path | None = None

    def reset(self) -> None:
        self.clips.clear()
        self.models.clear()
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

    def model(
        self,
        clip: PLCSMotionClip,
        *,
        model_root: Path,
        backend: PLCSExecutionBackend,
    ) -> object:
        if clip.gender not in self.models:
            self.models[clip.gender] = backend.load_model(
                model_root=model_root,
                gender=clip.gender,
            )
        return self.models[clip.gender]


@dataclass(frozen=True, slots=True)
class PLCSStageHandler:
    """Generate a balanced inventory of complete full-motion PLCS scenes."""

    configuration: PLCSDatasetConfiguration
    camera_configuration: CameraProfileConfig
    motion_library: ACCADMotionLibrary
    avatar_appearance_source: PLCSAvatarAppearanceSource
    renderer: NHTPLCSRenderer
    parameters: PLCSStageParameters
    execution_backend: PLCSExecutionBackend = field(
        default_factory=CUDAPLCSExecutionBackend,
        repr=False,
        compare=False,
    )
    _cache: _PLCSStageCache = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=_PLCSStageCache,
    )

    def preflight(self, context: StageExecutionContext) -> None:
        """Load each selected clip/gender once and validate the full inventory."""
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
        categories = set(self.configuration.motion_categories)
        requested_categories = {item.category.value for item in self.parameters.objects}
        if requested_categories != categories:
            raise ValueError(
                "PLCS requests must retain every config-authorized motion category."
            )
        if (
            self.parameters.scene_splits.get(context.request.scene_id)
            != self.parameters.split
        ):
            raise ValueError(
                "PLCS scene split disagrees with balanced assignment input."
            )
        self._cache.reset()
        self.execution_backend.reset_stage(
            configured_device=self.parameters.device,
            compositor=self.renderer.compositor,
        )
        _validate_execution_backend(
            self.execution_backend,
            configured_device=self.configuration.performance.execution_device,
            require_cuda=self.configuration.performance.require_cuda,
        )
        alignment = validate_alignment_outputs(paths.alignment_directory)
        assignments = assign_courts_balanced(
            self.parameters.scene_splits,
            layout=alignment.layout,
            seed=self.parameters.seed,
        )
        tracks = self._load_tracks_and_prepare_sources()
        _build_scene_inventory(
            dataset_scene_id=context.request.scene_id,
            assignments=assignments,
            layout=alignment.layout,
            tracks=tracks,
            required_motion_categories=frozenset(categories),
        )
        self.avatar_appearance_source.preflight(
            gaussian_count=self.parameters.gaussian_count,
        )

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        """Render every intact logical scene into one aggregate compact dataset."""
        timer = PerformanceTimer()
        paths = _stage_paths(context)
        if any(paths.staging_directory.iterdir()):
            raise ValueError(
                "PLCS stage requires an empty attempt-local staging directory."
            )
        if not self._cache.clips or not self._cache.models:
            raise RuntimeError(
                "PLCS execute requires its successful stage-scoped preflight cache."
            )
        alignment = validate_alignment_outputs(paths.alignment_directory)
        layout = alignment.layout
        assignments = assign_courts_balanced(
            self.parameters.scene_splits,
            layout=layout,
            seed=self.parameters.seed,
        )
        tracks = self._tracks_from_cache()
        inventory = _build_scene_inventory(
            dataset_scene_id=context.request.scene_id,
            assignments=assignments,
            layout=layout,
            tracks=tracks,
            required_motion_categories=frozenset(self.configuration.motion_categories),
        )
        rigs = self._sample_scene_rigs(inventory=inventory, layout=layout)
        unique_cameras = _unique_inventory_cameras(
            inventory=inventory,
            rigs=rigs,
        )
        resolutions = {(camera.width, camera.height) for camera in unique_cameras}
        if len(resolutions) != 1:
            raise ValueError("PLCS compact chunks require one configured resolution.")
        width, height = next(iter(resolutions))

        avatars = self._prepare_avatars(tracks)
        session = RenderSession(
            domain="plcs",
            attempt_token=f"{context.request.scene_id}-plcs",
            execution_device=self.execution_backend.execution_device,
        )
        self.renderer.render_background_store(
            scene_path=paths.scene_path,
            cameras=unique_cameras,
            metric_adapter=alignment.metric_adapter,
            staging_directory=paths.staging_directory,
            session=session,
        )
        for camera in unique_cameras:
            self.execution_backend.prepare_background(
                compositor=self.renderer.compositor,
                background=session.background("rig", camera.camera_id),
            )
        if self.execution_backend.background_upload_count != len(unique_cameras):
            raise RuntimeError(
                "PLCS did not prepare every static background exactly once."
            )

        scene_inputs: list[PLCSSceneAssemblyInput] = []
        chunk_size = self.configuration.timeline.chunk_size_frames
        writers: dict[str, ChunkWriter] = {}
        attempt_tokens: dict[str, str] = {}
        for logical in inventory.scenes:
            timeline = logical.timeline
            rig = rigs[timeline.scene_id]
            attempt_token = f"{context.request.scene_id}-plcs-{timeline.scene_id}"
            attempt_tokens[timeline.scene_id] = attempt_token
            writers[timeline.scene_id] = ChunkWriter(
                paths.staging_directory / "scenes" / timeline.scene_id / "chunks",
                attempt_token=attempt_token,
                camera_ids=tuple(
                    sampled.scene_camera.camera_id for sampled in rig.cameras
                ),
                width=width,
                height=height,
            )
        readers_by_scene, supervision_by_scene = self._render_logical_scenes(
            inventory=inventory,
            rigs=rigs,
            avatars=avatars,
            writers=writers,
            chunk_size=chunk_size,
        )
        for logical in inventory.scenes:
            timeline = logical.timeline
            scene_inputs.append(
                PLCSSceneAssemblyInput(
                    timeline=timeline,
                    split=logical.split,
                    rig=rigs[timeline.scene_id],
                    chunk_readers=readers_by_scene[timeline.scene_id],
                    attempt_token=attempt_tokens[timeline.scene_id],
                    supervision=supervision_by_scene[timeline.scene_id],
                )
            )

        supervision_bytes = sum(
            np.asarray(getattr(arrays, name)).nbytes
            for arrays in supervision_by_scene.values()
            for name in PLCSSupervisionArrays.__dataclass_fields__
        )
        transient_generated_bytes = (
            directory_size_bytes(paths.staging_directory) + supervision_bytes
        )
        _discard_background_working_files(paths.staging_directory)
        diagnostic_paths = write_plcs_diagnostics(
            staging_directory=paths.staging_directory,
            inventory=inventory,
            rigs=rigs,
            avatars=avatars,
            clip_load_count=len(self._cache.clips),
            model_load_count=len(self._cache.models),
            execution_device=self.execution_backend.execution_device,
            allow_test_cpu_oracle=(
                self.execution_backend.execution_device == "test-cpu-oracle"
            ),
        )
        all_diagnostics = (*diagnostic_paths, "diagnostics/performance.json")
        assembly = assemble_plcs_dataset(
            staging_directory=paths.staging_directory,
            inventory=inventory,
            scene_inputs=tuple(scene_inputs),
            chunk_size=chunk_size,
            diagnostics=all_diagnostics,
            seed=self.parameters.seed,
        )
        wall_seconds, cpu_seconds, peak_rss_bytes = timer.elapsed()
        sample_count = assembly.sample_count
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
            execution_device=self.execution_backend.execution_device,
            cuda_peak_bytes=self.execution_backend.cuda_peak_bytes,
            nht_invocations=session.nht_invocations,
            background_cache_misses=session.background_cache_misses,
            complete_array_scans=sample_count,
            generated_bytes=transient_generated_bytes,
            published_bytes=0,
            dense_reference_bytes=dense_reference_bytes,
            frame_count=inventory.aggregate_global_frame_count,
            camera_count=len(unique_cameras),
            sample_count=sample_count,
        )
        metrics = _write_performance_metrics(paths.staging_directory, metrics)
        metrics.validate_budget(self.configuration.performance)
        self._cache.validated_staging = paths.staging_directory
        return StageExecutionSummary(
            values={
                "logical_scene_count": inventory.scene_count,
                "source_motion_count_per_scene": len(tracks),
                "aggregate_source_frame_count": (
                    inventory.aggregate_source_frame_count
                ),
                "aggregate_global_frame_count": (
                    inventory.aggregate_global_frame_count
                ),
                "planned_frame_count": inventory.aggregate_global_frame_count,
                "rendered_frame_count": inventory.aggregate_global_frame_count,
                "labelled_frame_count": inventory.aggregate_global_frame_count,
                "camera_count": len(unique_cameras),
                "camera_count_per_scene": self.camera_configuration.expected_camera_count,
                "sample_count": assembly.sample_count,
                "chunk_count": assembly.chunk_count,
                "target_court_instance_ids": list(
                    inventory.accepted_court_instance_ids
                ),
                "continuity_record_count": assembly.continuity_record_count,
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
        alignment = validate_alignment_outputs(paths.alignment_directory)
        _validate_staged_court_inventory(
            paths.staging_directory,
            layout=alignment.layout,
        )

    def _load_tracks_and_prepare_sources(self) -> tuple[PLCSObjectTrack, ...]:
        tracks: list[PLCSObjectTrack] = []
        for index, request in enumerate(self.parameters.objects):
            clip = self._cache.clip(
                self.motion_library,
                request.category,
                seed=self.parameters.seed + index,
            )
            model = self._cache.model(
                clip,
                model_root=self.parameters.smplh_model_root,
                backend=self.execution_backend,
            )
            self.execution_backend.prepare_source(clip=clip, model=model)
            tracks.append(_object_track(index=index, request=request, clip=clip))
        return tuple(tracks)

    def _tracks_from_cache(self) -> tuple[PLCSObjectTrack, ...]:
        tracks = []
        for index, request in enumerate(self.parameters.objects):
            clip = self._cache.clip(
                self.motion_library,
                request.category,
                seed=self.parameters.seed + index,
            )
            tracks.append(_object_track(index=index, request=request, clip=clip))
        return tuple(tracks)

    def _prepare_avatars(
        self,
        tracks: tuple[PLCSObjectTrack, ...],
    ) -> dict[str, PLCSPreparedAvatar]:
        avatars: dict[str, PLCSPreparedAvatar] = {}
        for index, track in enumerate(tracks):
            appearance = self.avatar_appearance_source.load_avatar_appearance(
                clip=track.clip,
                object_id=track.object_id,
                gaussian_count=self.parameters.gaussian_count,
                seed=self.parameters.seed + index,
                device=self.execution_backend.torch_device,
            )
            avatars[track.object_id] = self.execution_backend.prepare_avatar(
                asset_id=track.asset_id,
                clip=track.clip,
                model=self._cache.models[track.clip.gender],
                appearance=appearance,
                gaussian_count=self.parameters.gaussian_count,
                seed=self.parameters.seed + index,
            )
        return avatars

    def _sample_scene_rigs(
        self,
        *,
        inventory: PLCSSceneInventory,
        layout: MultiCourtLayout,
    ) -> dict[str, SampledCameraRig]:
        rigs_by_court: dict[str, SampledCameraRig] = {}
        result: dict[str, SampledCameraRig] = {}
        projection_points = np.asarray(
            ((0.0, 0.0, 0.0), (-4.0, -10.0, 0.0), (4.0, 10.0, 1.0)),
            dtype=np.float64,
        )
        for logical in inventory.scenes:
            binding = logical.timeline.target_court
            court = layout.court(binding.court_instance_id)
            validate_court_transform_binding(
                layout,
                court_instance_id=binding.court_instance_id,
                candidate_id=binding.candidate_id,
                transforms={
                    "player_trajectory": binding.scene_from_court,
                    "generated_camera": court.scene_from_court,
                },
            )
            if binding.court_instance_id not in rigs_by_court:
                rig = sample_camera_rig(
                    self.camera_configuration,
                    seed=self.parameters.seed,
                    court=court,
                )
                for sampled in rig.cameras:
                    assert_projection_equivalent(
                        sampled,
                        court,
                        projection_points,
                        atol=1.0e-6,
                    )
                rigs_by_court[binding.court_instance_id] = rig
            result[logical.timeline.scene_id] = rigs_by_court[binding.court_instance_id]
        return result

    def _render_logical_scenes(
        self,
        *,
        inventory: PLCSSceneInventory,
        rigs: Mapping[str, SampledCameraRig],
        avatars: Mapping[str, PLCSPreparedAvatar],
        writers: Mapping[str, ChunkWriter],
        chunk_size: int,
    ) -> tuple[
        dict[str, tuple[ChunkReader, ...]],
        dict[str, PLCSSupervisionArrays],
    ]:
        """Render all courts together while evaluating each source batch once."""
        reference = inventory.scenes[0].timeline
        scene_ids = tuple(scene.timeline.scene_id for scene in inventory.scenes)
        if set(rigs) != set(scene_ids) or set(writers) != set(scene_ids):
            raise ValueError("PLCS render resources differ from its logical scenes.")
        compositions = {
            logical.timeline.scene_id: logical.timeline.to_foreground_composition(
                assets=tuple(
                    avatars[track.object_id].semantic_asset
                    for track in logical.timeline.tracks
                ),
            )
            for logical in inventory.scenes
        }
        supervision = {
            logical.timeline.scene_id: _empty_supervision(
                frame_count=logical.timeline.frame_count,
                camera_count=len(rigs[logical.timeline.scene_id].cameras),
                object_count=len(logical.timeline.tracks),
            )
            for logical in inventory.scenes
        }
        court_points = (
            court_keypoints_3d(STANDARD_COURT_CONFIG).numpy().astype(np.float64)
        )
        for logical in inventory.scenes[1:]:
            timeline = logical.timeline
            if timeline.frame_count != reference.frame_count or tuple(
                tuple(
                    (entry.object_id, entry.present, entry.source_frame_index)
                    for entry in frame.entries
                )
                for frame in timeline.frames
            ) != tuple(
                tuple(
                    (entry.object_id, entry.present, entry.source_frame_index)
                    for entry in frame.entries
                )
                for frame in reference.frames
            ):
                raise ValueError(
                    "PLCS logical scenes cannot share an inexact source timeline."
                )

        chunk_readers: dict[str, list[ChunkReader]] = {
            scene_id: [] for scene_id in scene_ids
        }
        for chunk_start in range(0, reference.frame_count, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, reference.frame_count)
            deltas: dict[str, list[ForegroundDelta]] = {
                scene_id: [] for scene_id in scene_ids
            }
            labels: dict[str, list[dict[str, object]]] = {
                scene_id: [] for scene_id in scene_ids
            }
            for batch_start in range(
                chunk_start,
                chunk_stop,
                self.parameters.smplh_batch_size,
            ):
                batch_stop = min(
                    batch_start + self.parameters.smplh_batch_size,
                    chunk_stop,
                )
                frame_tensors: dict[str, dict[int, PLCSAvatarFrameTensors]] = {}
                for track in reference.tracks:
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
                    for logical in inventory.scenes:
                        timeline = logical.timeline
                        scene_id = timeline.scene_id
                        rig = rigs[scene_id]
                        frame = timeline.frames[frame_index]
                        expected_ids = tuple(
                            entry.instance_id
                            for entry in frame.entries
                            if entry.present
                        )
                        object_tensors = {
                            entry.object_id: frame_tensors[entry.object_id][
                                int(entry.source_frame_index)
                            ].gaussians
                            for entry in frame.entries
                            if entry.present and entry.source_frame_index is not None
                        }
                        composed = compose_prevalidated_frame_gaussians(
                            compositions[scene_id],
                            frame_index=frame_index,
                            object_tensors=object_tensors,
                        )
                        _write_frame_supervision(
                            supervision[scene_id],
                            timeline=timeline,
                            rig=rig,
                            frame_index=frame_index,
                            frame_tensors=frame_tensors,
                            court_points_court_m=court_points,
                        )
                        for camera_index, sampled in enumerate(rig.cameras):
                            delta, visibility = self.execution_backend.compose_delta(
                                compositor=self.renderer.compositor,
                                frame_index=frame_index,
                                camera=sampled.scene_camera,
                                gaussians_scene=composed,
                                expected_instance_ids=expected_ids,
                            )
                            deltas[scene_id].append(delta)
                            labels[scene_id].append(
                                build_frame_label(
                                    timeline=timeline,
                                    rig=rig,
                                    frame_index=frame_index,
                                    camera_index=camera_index,
                                    visibility=visibility,
                                    seed=self.parameters.seed,
                                )
                            )
            for scene_id in scene_ids:
                chunk_readers[scene_id].append(
                    writers[scene_id].write(
                        ForegroundDeltaBatch(
                            chunk_id=f"chunk-{chunk_start:06d}",
                            deltas=tuple(deltas[scene_id]),
                            metadata=tuple(labels[scene_id]),
                        )
                    )
                )
        return (
            {scene_id: tuple(readers) for scene_id, readers in chunk_readers.items()},
            supervision,
        )


def _empty_supervision(
    *, frame_count: int, camera_count: int, object_count: int
) -> PLCSSupervisionArrays:
    """Allocate one exact dense semantic store with explicit inactive fills."""
    rotation: NDArray[np.float32] = np.zeros(
        (frame_count, object_count, 2), dtype=np.float32
    )
    rotation[..., 0] = 1.0
    return PLCSSupervisionArrays(
        human_kp=np.zeros(
            (frame_count, camera_count, object_count, 17, 2), dtype=np.float32
        ),
        human_vis=np.zeros(
            (frame_count, camera_count, object_count, 17), dtype=np.bool_
        ),
        court_kp=np.zeros((frame_count, camera_count, 20, 2), dtype=np.float32),
        court_vis=np.zeros((frame_count, camera_count, 20), dtype=np.bool_),
        human_mask=np.zeros((frame_count, camera_count, object_count), dtype=np.bool_),
        position=np.zeros((frame_count, object_count, 3), dtype=np.float32),
        position_court_m=np.zeros((frame_count, object_count, 3), dtype=np.float32),
        rotation=rotation,
        present=np.zeros((frame_count, object_count), dtype=np.bool_),
        human_kp_3d=np.zeros((frame_count, object_count, 17, 3), dtype=np.float32),
        canonical_pose_3d=np.zeros(
            (frame_count, object_count, 52, 3), dtype=np.float32
        ),
    )


def _write_frame_supervision(
    output: PLCSSupervisionArrays,
    *,
    timeline: PLCSGlobalTimeline,
    rig: SampledCameraRig,
    frame_index: int,
    frame_tensors: Mapping[str, Mapping[int, PLCSAvatarFrameTensors]],
    court_points_court_m: np.ndarray,
) -> None:
    """Project validated SMPL-H joints and court geometry through generated cameras."""
    frame = timeline.frames[frame_index]
    court_from_scene = timeline.target_court.scene_from_court.inverse()
    scene_court = timeline.target_court.scene_from_court.apply(court_points_court_m)
    scale = np.asarray(COURT_COORD_SCALE_XYZ, dtype=np.float32)
    present_joints: dict[int, np.ndarray] = {}
    for object_index, (track, entry) in enumerate(
        zip(timeline.tracks, frame.entries, strict=True)
    ):
        if track.object_id != entry.object_id:
            raise ValueError("PLCS frame entry order differs from its track authority.")
        if not entry.present:
            continue
        if entry.source_frame_index is None or entry.scene_from_asset is None:
            raise ValueError("Present PLCS entry lacks source/transform provenance.")
        local = (
            frame_tensors[entry.object_id][entry.source_frame_index]
            .joints_m.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        output.canonical_pose_3d[frame_index, object_index] = local
        joints_scene = entry.scene_from_asset.rigid.apply(
            local * entry.scene_from_asset.scale
        ).astype(np.float32)
        joints_court = court_from_scene.apply(joints_scene).astype(np.float32)
        coco = smplh_joints_to_coco17(joints_court[None], track.yaw_radians)[0]
        output.present[frame_index, object_index] = True
        output.human_mask[frame_index, :, object_index] = True
        output.position_court_m[frame_index, object_index] = joints_court[0]
        output.position[frame_index, object_index] = joints_court[0] / scale
        output.rotation[frame_index, object_index] = (
            np.cos(track.yaw_radians),
            np.sin(track.yaw_radians),
        )
        output.human_kp_3d[frame_index, object_index] = coco
        present_joints[object_index] = timeline.target_court.scene_from_court.apply(
            coco
        )

    for camera_index, sampled in enumerate(rig.cameras):
        camera = sampled.scene_camera
        court_pixels, court_depth = camera.project_scene_points(scene_court)
        court_visible = _inside_image(
            court_pixels, court_depth, width=camera.width, height=camera.height
        )
        output.court_kp[frame_index, camera_index] = _normalize_pixels(
            court_pixels, width=camera.width, height=camera.height
        )
        output.court_kp[frame_index, camera_index, ~court_visible] = 0.0
        output.court_vis[frame_index, camera_index] = court_visible
        for object_index, joints_scene in present_joints.items():
            pixels, depth = camera.project_scene_points(joints_scene)
            visible = _inside_image(
                pixels, depth, width=camera.width, height=camera.height
            )
            output.human_kp[frame_index, camera_index, object_index] = (
                _normalize_pixels(pixels, width=camera.width, height=camera.height)
            )
            output.human_kp[frame_index, camera_index, object_index, ~visible] = 0.0
            output.human_vis[frame_index, camera_index, object_index] = visible


def _inside_image(
    pixels: np.ndarray,
    depth: np.ndarray,
    *,
    width: int,
    height: int,
) -> NDArray[np.bool_]:
    return np.asarray(
        (depth > 0.0)
        & (pixels[..., 0] >= 0.0)
        & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0.0)
        & (pixels[..., 1] < height),
        dtype=np.bool_,
    )


def _normalize_pixels(
    pixels: np.ndarray, *, width: int, height: int
) -> NDArray[np.float32]:
    result: NDArray[np.float32] = pixels.astype(np.float32, copy=True)
    result[..., 0] /= width
    result[..., 1] /= height
    return result


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


def _build_scene_inventory(
    *,
    dataset_scene_id: str,
    assignments: tuple[CourtAssignment, ...],
    layout: MultiCourtLayout,
    tracks: tuple[PLCSObjectTrack, ...],
    required_motion_categories: frozenset[str],
) -> PLCSSceneInventory:
    scenes = []
    for assignment in assignments:
        court = layout.court(assignment.court_instance_id)
        binding = TargetCourtBinding(
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            scene_from_court=court.scene_from_court,
            selection_seed=assignment.selection_seed,
        )
        scenes.append(
            PLCSLogicalScene(
                split=assignment.split,
                timeline=build_global_timeline(
                    scene_id=assignment.scene_id,
                    target_court=binding,
                    tracks=tracks,
                ),
            )
        )
    return PLCSSceneInventory(
        dataset_scene_id=dataset_scene_id,
        scenes=tuple(scenes),
        accepted_court_instance_ids=tuple(
            court.court_instance_id for court in layout.courts
        ),
        required_motion_categories=required_motion_categories,
    )


def _unique_inventory_cameras(
    *,
    inventory: PLCSSceneInventory,
    rigs: Mapping[str, SampledCameraRig],
) -> tuple[SceneCamera, ...]:
    cameras: list[SceneCamera] = []
    seen_courts: set[str] = set()
    seen_camera_ids: set[str] = set()
    for logical in inventory.scenes:
        court_id = logical.timeline.target_court.court_instance_id
        if court_id in seen_courts:
            continue
        seen_courts.add(court_id)
        for sampled in rigs[logical.timeline.scene_id].cameras:
            camera = sampled.scene_camera
            if camera.camera_id in seen_camera_ids:
                raise ValueError("PLCS accepted-court camera IDs must be unique.")
            seen_camera_ids.add(camera.camera_id)
            cameras.append(camera)
    if seen_courts != set(inventory.accepted_court_instance_ids):
        raise ValueError("PLCS camera rigs do not cover every accepted court.")
    return tuple(cameras)


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


def _validate_execution_backend(
    backend: PLCSExecutionBackend,
    *,
    configured_device: str,
    require_cuda: bool,
) -> None:
    if backend.execution_device != configured_device:
        raise ValueError(
            "PLCS execution backend differs from config-owned performance authority."
        )
    if backend.execution_device == "test-cpu-oracle":
        if require_cuda or backend.torch_device.type != "cpu":
            raise ValueError(
                "PLCS test CPU oracle requires an explicit non-CUDA test budget."
            )
    elif (
        not require_cuda
        or not backend.execution_device.startswith("cuda")
        or backend.torch_device.type != "cuda"
    ):
        raise ValueError("PLCS production execution requires explicit CUDA.")


def _validate_staged_court_inventory(
    staging: Path,
    *,
    layout: MultiCourtLayout,
) -> None:
    manifest_path = staging / "dataset.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("PLCS staged dataset manifest is missing.")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("PLCS staged dataset manifest must be a JSON object.")
    metadata = raw.get("metadata")
    target_courts = raw.get("target_courts")
    if not isinstance(metadata, dict) or not isinstance(target_courts, list):
        raise TypeError("PLCS staged court metadata is malformed.")
    accepted_ids = [court.court_instance_id for court in layout.courts]
    if metadata.get("accepted_court_instance_ids") != accepted_ids:
        raise ValueError(
            "PLCS staged dataset does not cover the accepted alignment courts."
        )
    bindings = tuple(TargetCourtBinding.from_dict(value) for value in target_courts)
    if [binding.court_instance_id for binding in bindings] != accepted_ids:
        raise ValueError("PLCS staged target-court binding inventory is incomplete.")
    for binding in bindings:
        court = layout.court(binding.court_instance_id)
        if (
            binding.candidate_id != court.candidate_id
            or binding.scene_from_court != court.scene_from_court
        ):
            raise ValueError(
                "PLCS staged target-court geometry differs from alignment."
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
