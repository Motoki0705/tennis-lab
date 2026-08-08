"""One-background-per-trajectory BLCS rendering with compact CUDA deltas."""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCS_SAMPLE_SCHEMA,
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.dataset.blcs.timeline import BLCSTrajectoryPlan
from src.synthetic_data_generation.dataset.runtime import (
    BackgroundArrays,
    ChunkReader,
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
    RenderSession,
    directory_size_bytes,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderClient
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderCamera,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)

_BALL_RGB = (1.0, 0.85, 0.0)


class BLCSForegroundCompositor(Protocol):
    """Explicit compact-foreground compositor boundary.

    Production supplies :class:`CUDABLCSForegroundCompositor`. Tests may inject
    an independently implemented oracle whose ``execution_device`` is exactly
    ``"test-cpu-oracle"``; there is no automatic CPU selection.
    """

    @property
    def execution_device(self) -> str:
        """Return the exact evidence device identifier."""

    @property
    def cuda_peak_bytes(self) -> int:
        """Return peak CUDA allocation observed by this compositor."""

    def compose(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        backgrounds: Mapping[str, BackgroundArrays],
        ball_radius_m: float,
    ) -> Iterator[ForegroundDeltaBatch]:
        """Yield one compact batch per resolved BLCS chunk."""


@dataclass(slots=True)
class CUDABLCSForegroundCompositor:
    """Rasterize sparse ball discs on one required CUDA device.

    Background depth stays resident once per trajectory-camera. Composition is
    bounded by ``maximum_batch_frames`` and only sparse foreground values cross
    back to host memory.
    """

    device: str
    maximum_batch_frames: int
    _cuda_peak_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.device.startswith("cuda"):
            raise ValueError("Production BLCS composition requires a CUDA device.")
        if (
            isinstance(self.maximum_batch_frames, bool)
            or not isinstance(self.maximum_batch_frames, int)
            or self.maximum_batch_frames <= 0
        ):
            raise ValueError("maximum_batch_frames must be a positive integer.")

    @property
    def execution_device(self) -> str:
        """Return the configured CUDA device."""
        return self.device

    @property
    def cuda_peak_bytes(self) -> int:
        """Return the maximum allocated CUDA bytes observed so far."""
        return self._cuda_peak_bytes

    def compose(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        backgrounds: Mapping[str, BackgroundArrays],
        ball_radius_m: float,
    ) -> Iterator[ForegroundDeltaBatch]:
        """Compose every resolved chunk without allocating dense output frames."""
        if not torch.cuda.is_available():
            raise RuntimeError(
                "BLCS production composition requires CUDA; no CPU fallback exists."
            )
        device = torch.device(self.device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        camera_ids = tuple(
            sampled.scene_camera.camera_id for sampled in plan.camera_rig.cameras
        )
        if tuple(backgrounds) != camera_ids:
            raise ValueError("BLCS background cache differs from camera-rig order.")
        device_depth = {
            camera_id: torch.tensor(
                backgrounds[camera_id].depth.reshape(-1),
                dtype=torch.float32,
                device=device,
            )
            for camera_id in camera_ids
        }
        try:
            for chunk in plan.chunks:
                deltas: list[ForegroundDelta] = []
                metadata: list[Mapping[str, object]] = []
                frame_indices = chunk.frame_indices
                for batch_start in range(
                    0, len(frame_indices), self.maximum_batch_frames
                ):
                    batch_frames = frame_indices[
                        batch_start : batch_start + self.maximum_batch_frames
                    ]
                    for source_frame_index in batch_frames:
                        for camera_index, camera_id in enumerate(camera_ids):
                            delta = _compose_cuda_delta(
                                plan=plan,
                                source_frame_index=source_frame_index,
                                camera_index=camera_index,
                                background=backgrounds[camera_id],
                                background_depth=device_depth[camera_id],
                                ball_radius_m=ball_radius_m,
                                device=device,
                            )
                            deltas.append(delta)
                            metadata.append(
                                build_blcs_sample_metadata(
                                    plan=plan,
                                    source_frame_index=source_frame_index,
                                    camera_index=camera_index,
                                    chunk_index=chunk.chunk_index,
                                    delta=delta,
                                )
                            )
                torch.cuda.synchronize(device)
                self._cuda_peak_bytes = max(
                    self._cuda_peak_bytes,
                    int(torch.cuda.max_memory_allocated(device)),
                )
                yield ForegroundDeltaBatch(
                    chunk_id=f"chunk-{chunk.chunk_index:06d}",
                    deltas=tuple(deltas),
                    metadata=tuple(metadata),
                )
        finally:
            device_depth.clear()


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
    """Invoke public NHT exactly once per trajectory, then write compact chunks."""

    assets: BLCSCompositionAssets
    client: NHTRenderClient
    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float
    execution_device: str
    maximum_batch_frames: int
    test_cpu_oracle: BLCSForegroundCompositor | None = None

    def __post_init__(self) -> None:
        if self.test_cpu_oracle is not None:
            if self.test_cpu_oracle.execution_device != "test-cpu-oracle":
                raise ValueError(
                    "An injected BLCS test oracle must identify itself as test-cpu-oracle."
                )
        elif not self.execution_device.startswith("cuda"):
            raise ValueError("BLCS production renderer requires explicit CUDA.")
        if (
            isinstance(self.maximum_batch_frames, bool)
            or not isinstance(self.maximum_batch_frames, int)
            or self.maximum_batch_frames <= 0
        ):
            raise ValueError("maximum_batch_frames must be a positive integer.")

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
        compositor: BLCSForegroundCompositor
        if self.test_cpu_oracle is not None:
            compositor = self.test_cpu_oracle
        else:
            compositor = CUDABLCSForegroundCompositor(
                device=self.execution_device,
                maximum_batch_frames=self.maximum_batch_frames,
            )
        session = RenderSession(
            domain="blcs",
            attempt_token=attempt_token,
            execution_device=compositor.execution_device,
        )
        rendered_trajectories: list[BLCSRenderedTrajectory] = []
        generated_bytes = 0
        for plan in plan_tuple:
            rendered, trajectory_generated = self._render_trajectory(
                plan=plan,
                scene_path=scene_path,
                samples_directory=samples_directory,
                metric_adapter=metric_adapter,
                session=session,
                compositor=compositor,
            )
            rendered_trajectories.append(rendered)
            generated_bytes += trajectory_generated
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
            execution_device=compositor.execution_device,
            cuda_peak_bytes=compositor.cuda_peak_bytes,
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
        compositor: BLCSForegroundCompositor,
    ) -> tuple[BLCSRenderedTrajectory, int]:
        trajectory_directory = samples_directory / plan.source.trajectory_id
        trajectory_directory.mkdir(parents=False, exist_ok=False)
        request = _nht_request(plan=plan, metric_adapter=metric_adapter)
        request_path = trajectory_directory / "nht-cameras.json"
        working_background = trajectory_directory / "nht-background-working"
        command = NHTRenderCommandRequest(
            scene_path=scene_path,
            output_directory=working_background,
            arbitrary_cameras=request,
            arbitrary_request_path=request_path,
            executable=self.executable,
        )
        session.note_nht_invocation()
        result = self.client.render(
            command,
            environment=self.environment,
            timeout_seconds=self.timeout_seconds,
        )
        if result.scene_id != plan.dataset_scene_id:
            raise ValueError("NHT render result scene_id disagrees with the BLCS plan.")
        nht_generated_bytes = directory_size_bytes(working_background) + (
            request_path.stat().st_size if request_path.is_file() else 0
        )
        camera_ids = tuple(
            sampled.scene_camera.camera_id for sampled in plan.camera_rig.cameras
        )
        background_directory = trajectory_directory / "backgrounds"
        session.create_background_store(
            plan.source.trajectory_id,
            background_directory,
            rendered=result,
            nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
            expected_camera_ids=camera_ids,
        )
        shutil.rmtree(working_background)
        request_path.unlink(missing_ok=True)
        backgrounds = {
            camera_id: session.background(plan.source.trajectory_id, camera_id)
            for camera_id in camera_ids
        }
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
        for batch in compositor.compose(
            plan=plan,
            backgrounds=backgrounds,
            ball_radius_m=self.assets.ball_radius_m,
        ):
            rendered_visible_object_views += sum(
                len(delta.visible_instance_counts) for delta in batch.deltas
            )
            chunk_readers.append(writer.write(batch))
        readers = tuple(chunk_readers)
        if tuple(reader.directory.name for reader in readers) != tuple(
            f"chunk-{chunk.chunk_index:06d}" for chunk in plan.chunks
        ):
            raise ValueError(
                "BLCS compositor did not emit one marker per resolved chunk."
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
        )


def _compose_cuda_delta(
    *,
    plan: BLCSTrajectoryPlan,
    source_frame_index: int,
    camera_index: int,
    background: BackgroundArrays,
    background_depth: Tensor,
    ball_radius_m: float,
    device: torch.device,
) -> ForegroundDelta:
    camera = plan.camera_rig.cameras[camera_index].scene_camera
    focal = float(np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)[0, 0])
    pixel_parts: list[Tensor] = []
    depth_parts: list[Tensor] = []
    id_parts: list[Tensor] = []
    for object_index in range(plan.source.object_count):
        if not plan.geometric_visible[source_frame_index, camera_index, object_index]:
            continue
        centre = plan.camera_uv[source_frame_index, camera_index, object_index]
        object_depth = float(
            plan.camera_depth[source_frame_index, camera_index, object_index]
        )
        radius_pixels = max(1, int(round(focal * ball_radius_m / object_depth)))
        x_min = max(0, int(math.floor(centre[0] - radius_pixels)))
        x_max = min(camera.width, int(math.ceil(centre[0] + radius_pixels + 1)))
        y_min = max(0, int(math.floor(centre[1] - radius_pixels)))
        y_max = min(camera.height, int(math.ceil(centre[1] + radius_pixels + 1)))
        xs = torch.arange(x_min, x_max, dtype=torch.int64, device=device)
        ys = torch.arange(y_min, y_max, dtype=torch.int64, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centre_tensor = torch.tensor(centre, dtype=torch.float64, device=device)
        disc = (xx.to(torch.float64) - centre_tensor[0]).square() + (
            yy.to(torch.float64) - centre_tensor[1]
        ).square() <= radius_pixels**2
        pixels = (yy * camera.width + xx)[disc]
        if pixels.numel() == 0:
            continue
        visible = (background_depth[pixels] <= 0.0) | (
            object_depth <= background_depth[pixels]
        )
        pixels = pixels[visible]
        if pixels.numel() == 0:
            continue
        pixel_parts.append(pixels)
        depth_parts.append(
            torch.full(
                (pixels.numel(),), object_depth, dtype=torch.float32, device=device
            )
        )
        id_parts.append(
            torch.full(
                (pixels.numel(),),
                object_index + 1,
                dtype=torch.int32,
                device=device,
            )
        )
    if pixel_parts:
        pixels = torch.cat(pixel_parts)
        depths = torch.cat(depth_parts)
        instance_ids = torch.cat(id_parts)
        unique_pixels, inverse = torch.unique(pixels, sorted=True, return_inverse=True)
        minimum_depth = torch.full(
            (unique_pixels.numel(),),
            torch.inf,
            dtype=torch.float32,
            device=device,
        )
        minimum_depth.scatter_reduce_(
            0, inverse, depths, reduce="amin", include_self=True
        )
        is_nearest = depths == minimum_depth[inverse]
        nearest_ids = torch.where(
            is_nearest,
            instance_ids,
            torch.zeros_like(instance_ids),
        )
        selected_ids = torch.zeros(
            (unique_pixels.numel(),),
            dtype=torch.int32,
            device=device,
        )
        selected_ids.scatter_reduce_(
            0, inverse, nearest_ids, reduce="amax", include_self=True
        )
        rgb = torch.tensor(_BALL_RGB, dtype=torch.float32, device=device).expand(
            unique_pixels.numel(), 3
        )
        alpha = torch.ones(unique_pixels.numel(), dtype=torch.float32, device=device)
    else:
        unique_pixels = torch.empty(0, dtype=torch.int64, device=device)
        rgb = torch.empty((0, 3), dtype=torch.float32, device=device)
        alpha = torch.empty(0, dtype=torch.float32, device=device)
        minimum_depth = torch.empty(0, dtype=torch.float32, device=device)
        selected_ids = torch.empty(0, dtype=torch.int32, device=device)
    return ForegroundDelta(
        key=RenderSampleKey(source_frame_index, camera.camera_id),
        pixel_indices=unique_pixels.to(dtype=torch.int32).cpu().numpy(),
        rgb=rgb.contiguous().cpu().numpy(),
        alpha=alpha.cpu().numpy(),
        depth=minimum_depth.cpu().numpy(),
        instance_ids=selected_ids.cpu().numpy(),
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
    "BLCSForegroundCompositor",
    "BLCSNHTRenderer",
    "BLCSRenderAttempt",
    "BLCSRenderedTrajectory",
    "CUDABLCSForegroundCompositor",
    "build_blcs_sample_metadata",
]
