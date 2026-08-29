"""Joint NHT Gaussian rendering for compact BLCS datasets."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.ball_asset import (
    build_ball_gaussian_asset,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCS_SAMPLE_SCHEMA,
    BLCSBallRendering,
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.dataset.blcs.rendering.request import (
    write_blcs_nht_composition_request,
)
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan
from src.synthetic_data_generation.dataset.runtime import (
    ChunkReader,
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
    RenderSession,
    directory_size_bytes,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTComposedChunkRecord,
    NHTComposedRenderClient,
    NHTComposedRenderCommandRequest,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.rendering.nht.depth import nht_depth_to_metric


@dataclass(frozen=True, slots=True)
class BLCSRenderedTrajectory:
    """Current-attempt compact files for one source trajectory."""

    trajectory_id: str
    directory: Path
    background_directory: Path
    chunk_readers: tuple[ChunkReader, ...]
    rendered_visible_object_views: int


@dataclass(frozen=True, slots=True)
class BLCSRenderAttempt:
    """Measured stage-local output returned to the BLCS assembler."""

    attempt_token: str
    trajectories: tuple[BLCSRenderedTrajectory, ...]
    execution_device: str
    cuda_peak_bytes: int
    nht_invocations: int
    background_cache_misses: int
    generated_bytes: int


@dataclass(frozen=True, slots=True)
class BLCSNHTRenderer:
    """Jointly rasterize the NHT scene and moving Gaussian balls once per trajectory."""

    assets: BLCSCompositionAssets
    client: NHTComposedRenderClient
    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float
    execution_device: str
    maximum_batch_frames: int

    def __post_init__(self) -> None:
        if self.assets.rendering is not BLCSBallRendering.GAUSSIAN:
            raise ValueError("BLCSNHTRenderer requires assets.rendering=gaussian.")
        if not isinstance(self.client, NHTComposedRenderClient):
            raise TypeError("BLCS renderer requires NHTComposedRenderClient.")
        if self.execution_device != "cuda:0":
            raise ValueError("The public composed NHT renderer currently owns cuda:0.")
        if self.maximum_batch_frames != 1:
            raise ValueError("The public composed NHT renderer rasterizes one frame at a time.")

    def validate_asset(self) -> None:
        """Build and validate the complete Gaussian asset before stage invalidation."""
        build_ball_gaussian_asset(self.assets)

    def render(
        self,
        *,
        plans: Sequence[BLCSTrajectoryPlan],
        scene_path: Path,
        samples_directory: Path,
        metric_adapter: MetricSceneAdapter,
        attempt_token: str,
    ) -> BLCSRenderAttempt:
        """Render a complete BLCS attempt without a per-chunk NHT path."""
        plan_tuple = tuple(plans)
        if not plan_tuple:
            raise ValueError("BLCS rendering requires at least one trajectory plan.")
        if samples_directory.exists() or samples_directory.is_symlink():
            raise FileExistsError(
                "BLCS compact samples must start from an empty attempt directory."
            )
        samples_directory.mkdir(parents=True, exist_ok=False)
        session = RenderSession(
            domain="blcs",
            attempt_token=attempt_token,
            execution_device=self.execution_device,
        )
        rendered_trajectories: list[BLCSRenderedTrajectory] = []
        generated_bytes = 0
        cuda_peak_bytes = 0
        for plan in plan_tuple:
            rendered, trajectory_generated, trajectory_cuda_peak = self._render_trajectory(
                plan=plan,
                scene_path=scene_path,
                samples_directory=samples_directory,
                metric_adapter=metric_adapter,
                session=session,
            )
            rendered_trajectories.append(rendered)
            generated_bytes += trajectory_generated
            cuda_peak_bytes = max(cuda_peak_bytes, trajectory_cuda_peak)
        expected_misses = sum(len(plan.camera_rig.cameras) for plan in plan_tuple)
        if session.nht_invocations != len(plan_tuple):
            raise ValueError("BLCS must invoke NHT exactly once per trajectory.")
        if session.background_cache_misses != expected_misses:
            raise ValueError(
                "BLCS must validate exactly one background cache miss per trajectory-camera."
            )
        return BLCSRenderAttempt(
            attempt_token=attempt_token,
            trajectories=tuple(rendered_trajectories),
            execution_device=self.execution_device,
            cuda_peak_bytes=cuda_peak_bytes,
            nht_invocations=session.nht_invocations,
            background_cache_misses=session.background_cache_misses,
            generated_bytes=generated_bytes,
        )

    def _render_trajectory(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        scene_path: Path,
        samples_directory: Path,
        metric_adapter: MetricSceneAdapter,
        session: RenderSession,
    ) -> tuple[BLCSRenderedTrajectory, int, int]:
        trajectory_directory = samples_directory / plan.source.trajectory_id
        trajectory_directory.mkdir(parents=False, exist_ok=False)
        request = _nht_request(plan=plan, metric_adapter=metric_adapter)
        request_path = trajectory_directory / "nht-cameras.json"
        composition_directory = trajectory_directory / "nht-composition-request"
        working_render = trajectory_directory / "nht-composed-working"
        with ExitStack() as temporary:
            temporary.callback(_remove_temporary_directory, composition_directory)
            temporary.callback(_remove_temporary_directory, working_render)
            temporary.callback(request_path.unlink, missing_ok=True)
            composition_files = write_blcs_nht_composition_request(
                composition_directory,
                plan=plan,
                assets=self.assets,
                metric_adapter=metric_adapter,
            )
            base_command = NHTRenderCommandRequest(
                scene_path=scene_path,
                output_directory=working_render,
                arbitrary_cameras=request,
                arbitrary_request_path=request_path,
                executable=self.executable,
            )
            command = NHTComposedRenderCommandRequest(
                base=base_command,
                composition_request_path=composition_files.request_path.resolve(
                    strict=True
                ),
            )
            session.note_nht_invocation()
            result = self.client.render_composed(
                command,
                environment=dict(self.environment),
                timeout_seconds=self.timeout_seconds,
            )
            if result.scene_id != plan.dataset_scene_id:
                raise ValueError(
                    "Composed NHT result scene_id disagrees with the BLCS plan."
                )
            nht_generated_bytes = (
                directory_size_bytes(working_render)
                + directory_size_bytes(composition_files.request_path.parent)
                + (request_path.stat().st_size if request_path.is_file() else 0)
            )
            camera_ids = tuple(
                sampled.scene_camera.camera_id for sampled in plan.camera_rig.cameras
            )
            background_directory = trajectory_directory / "backgrounds"
            session.create_background_store(
                plan.source.trajectory_id,
                background_directory,
                rendered=result.background,
                nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
                expected_camera_ids=camera_ids,
            )
            first_camera = plan.camera_rig.cameras[0].scene_camera
            if any(
                sampled.scene_camera.width != first_camera.width
                or sampled.scene_camera.height != first_camera.height
                for sampled in plan.camera_rig.cameras
            ):
                raise ValueError(
                    "BLCS compact chunks require one camera image shape per rig."
                )
            writer = ChunkWriter(
                trajectory_directory / "chunks",
                attempt_token=session.attempt_token,
                camera_ids=camera_ids,
                width=first_camera.width,
                height=first_camera.height,
            )
            chunk_readers: list[ChunkReader] = []
            rendered_visible_object_views = 0
            if tuple(record.chunk_id for record in result.chunks) != tuple(
                f"chunk-{chunk.chunk_index:06d}" for chunk in plan.chunks
            ):
                raise ValueError(
                    "Composed NHT chunk inventory differs from the BLCS plan."
                )
            for record in result.chunks:
                batch = _foreground_batch_from_composed_record(
                    plan=plan,
                    record=record,
                    nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
                )
                rendered_visible_object_views += sum(
                    len(delta.visible_instance_counts) for delta in batch.deltas
                )
                chunk_readers.append(writer.write(batch))
            readers = tuple(chunk_readers)
            if tuple(reader.directory.name for reader in readers) != tuple(
                f"chunk-{chunk.chunk_index:06d}" for chunk in plan.chunks
            ):
                raise ValueError(
                    "BLCS joint renderer did not emit one marker per resolved chunk."
                )
            canonical_bytes = directory_size_bytes(
                background_directory
            ) + directory_size_bytes(trajectory_directory / "chunks")
            return (
                BLCSRenderedTrajectory(
                    trajectory_id=plan.source.trajectory_id,
                    directory=trajectory_directory,
                    background_directory=background_directory,
                    chunk_readers=readers,
                    rendered_visible_object_views=rendered_visible_object_views,
                ),
                nht_generated_bytes + canonical_bytes,
                result.cuda_peak_bytes,
            )


def _remove_temporary_directory(path: Path) -> None:
    """Remove only one renderer-owned temporary path without following symlinks."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _foreground_batch_from_composed_record(
    *,
    plan: BLCSTrajectoryPlan,
    record: NHTComposedChunkRecord,
    nht_scene_units_per_metre: float,
) -> ForegroundDeltaBatch:
    """Transfer one validated joint-raster result into the compact dataset contract."""
    arrays = record.load_arrays()
    metric_depth = nht_depth_to_metric(
        arrays.depth,
        nht_scene_units_per_metre=nht_scene_units_per_metre,
    )
    deltas: list[ForegroundDelta] = []
    metadata: list[Mapping[str, object]] = []
    for sample_index in range(record.sample_count):
        source_frame_index = int(arrays.frame_indices[sample_index])
        camera_index = int(arrays.camera_indices[sample_index])
        camera_id = record.camera_ids[camera_index]
        start = int(arrays.offsets[sample_index])
        stop = int(arrays.offsets[sample_index + 1])
        delta = ForegroundDelta(
            key=RenderSampleKey(source_frame_index, camera_id),
            pixel_indices=arrays.pixel_indices[start:stop],
            rgb=arrays.rgb[start:stop],
            alpha=arrays.alpha[start:stop],
            depth=metric_depth[start:stop],
            instance_ids=arrays.instance_ids[start:stop],
        )
        deltas.append(delta)
        metadata.append(
            build_blcs_sample_metadata(
                plan=plan,
                source_frame_index=source_frame_index,
                camera_index=camera_index,
                chunk_index=int(record.chunk_id.removeprefix("chunk-")),
                delta=delta,
            )
        )
    return ForegroundDeltaBatch(
        chunk_id=record.chunk_id,
        deltas=tuple(deltas),
        metadata=tuple(metadata),
    )


def build_blcs_sample_metadata(
    *,
    plan: BLCSTrajectoryPlan,
    source_frame_index: int,
    camera_index: int,
    chunk_index: int,
    delta: ForegroundDelta,
) -> dict[str, object]:
    camera = plan.camera_rig.cameras[camera_index]
    rendered_ids = set(int(value) for value in np.unique(delta.instance_ids))
    source_indices = [
        track.source_frame_indices[source_frame_index] for track in plan.source.tracks
    ]
    rendered_visible = [
        object_index + 1 in rendered_ids
        for object_index in range(plan.source.object_count)
    ]
    objects = [
        {
            "object_id": track.object_id,
            "instance_id": object_index + 1,
            "present": bool(plan.source.present[source_frame_index, object_index]),
            "source_trajectory": track.source_trajectory_id,
            "source_frame": source_indices[object_index],
            "geometric_visible": bool(
                plan.geometric_visible[source_frame_index, camera_index, object_index]
            ),
            "rendered_visible": rendered_visible[object_index],
        }
        for object_index, track in enumerate(plan.source.tracks)
    ]
    return {
        "schema": BLCS_SAMPLE_SCHEMA,
        "scene_id": plan.dataset_scene_id,
        "trajectory_id": plan.source.trajectory_id,
        "split": plan.source.split,
        "global_frame_index": plan.global_frame_offset + source_frame_index,
        "source_frame_index": source_frame_index,
        "chunk_index": chunk_index,
        "source_trajectory": plan.source.trajectory_id,
        "source_frame": source_frame_index,
        "target_court": plan.target_court.court_instance_id,
        "candidate_id": plan.target_court.candidate_id,
        "transform": plan.target_court.scene_from_court.to_list(),
        "camera_profile": plan.camera_rig.profile,
        "camera_parameters": camera.to_metadata(),
        "seed": {
            "court_assignment": plan.target_court.selection_seed,
            "camera_sampling": plan.camera_rig.seed,
        },
        "objects": objects,
        "semantic_arrays": {
            "ball_uv": plan.camera_uv[source_frame_index, camera_index].tolist(),
            "ball_depth": plan.camera_depth[source_frame_index, camera_index].tolist(),
            "geometric_visible": plan.geometric_visible[
                source_frame_index, camera_index
            ].tolist(),
            "rendered_visible": rendered_visible,
            "positions_court_m": plan.source.positions_court_m[
                source_frame_index
            ].tolist(),
            "positions_scene": plan.positions_scene[source_frame_index].tolist(),
            "velocities_court_mps": plan.source.velocities_court_mps[
                source_frame_index
            ].tolist(),
            "present": plan.source.present[source_frame_index].tolist(),
            "source_frame_indices": source_indices,
            "instance_ids": list(range(1, plan.source.object_count + 1)),
        },
    }


def _nht_request(
    *,
    plan: BLCSTrajectoryPlan,
    metric_adapter: MetricSceneAdapter,
) -> NHTRenderRequest:
    if not isinstance(metric_adapter, MetricSceneAdapter):
        raise TypeError("BLCS renderer requires the accepted MetricSceneAdapter.")
    cameras: list[NHTRenderCamera] = []
    for sampled in plan.camera_rig.cameras:
        metric_camera = sampled.scene_camera
        nht_pose = metric_adapter.nht_from_metric_camera(metric_camera.camera_to_scene)
        round_trip = metric_adapter.metric_from_nht_camera(nht_pose)
        if not np.allclose(
            round_trip.matrix(),
            metric_camera.camera_to_scene.matrix(),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "BLCS metric/NHT arbitrary-camera pose round trip is inconsistent."
            )
        cameras.append(
            NHTRenderCamera.from_scene_camera(
                replace(metric_camera, camera_to_scene=nht_pose)
            )
        )
    return NHTRenderRequest(cameras=tuple(cameras))


__all__ = [
    "BLCSNHTRenderer",
    "BLCSRenderAttempt",
    "BLCSRenderedTrajectory",
    "build_blcs_sample_metadata",
]
