"""Depth-correct GLB ball rendering over public NHT 3DGS backgrounds."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.synthetic_data_generation.alignment import MetricSceneAdapter
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSBallRendering,
    BLCSCompositionAssets,
)
from src.synthetic_data_generation.dataset.blcs.mesh_asset import (
    BLCSBallMesh,
    load_ball_mesh_asset,
)
from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    BLCSRenderAttempt,
    BLCSRenderedTrajectory,
    _nht_request,
    build_blcs_sample_metadata,
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
    NHTRenderCommandRequest,
)

_RAY_TRIANGLE_CHUNK = 1024


@dataclass(frozen=True, slots=True)
class BLCSMeshRasterizer:
    """Ray-cast a bounded triangle mesh into exact camera-axis metric depth."""

    mesh: BLCSBallMesh
    device: str
    ambient: float = 0.42
    diffuse: float = 0.58
    _vertices: Tensor = field(init=False, repr=False, compare=False)
    _faces: Tensor = field(init=False, repr=False, compare=False)
    _normals: Tensor = field(init=False, repr=False, compare=False)
    _colors: Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, BLCSBallMesh):
            raise TypeError("mesh must be BLCSBallMesh.")
        try:
            device = torch.device(self.device)
        except RuntimeError as error:
            raise ValueError(
                f"Invalid BLCS mesh rendering device: {self.device!r}."
            ) from error
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "BLCS mesh rasterization supports only cpu or cuda devices."
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "BLCS mesh rendering requested CUDA, but CUDA is unavailable."
            )
        if not 0.0 <= self.ambient <= 1.0 or not 0.0 <= self.diffuse <= 1.0:
            raise ValueError("BLCS mesh lighting terms must be in [0,1].")
        if self.ambient + self.diffuse > 1.0 + 1.0e-8:
            raise ValueError("BLCS mesh ambient + diffuse must not exceed 1.")
        object.__setattr__(
            self,
            "_vertices",
            torch.tensor(self.mesh.vertices_m, device=device, dtype=torch.float32),
        )
        object.__setattr__(
            self,
            "_faces",
            torch.tensor(self.mesh.faces, device=device, dtype=torch.int64),
        )
        object.__setattr__(
            self,
            "_normals",
            torch.tensor(self.mesh.normals, device=device, dtype=torch.float32),
        )
        object.__setattr__(
            self,
            "_colors",
            torch.tensor(
                self.mesh.colors_linear_rgb,
                device=device,
                dtype=torch.float32,
            ),
        )

    def render_sample(
        self,
        *,
        plan: BLCSTrajectoryPlan,
        source_frame_index: int,
        camera_index: int,
        background: BackgroundArrays,
    ) -> ForegroundDelta:
        """Render all present balls with mesh/mesh and mesh/background z-buffering."""
        if not 0 <= source_frame_index < plan.source.frame_count:
            raise IndexError("BLCS mesh source_frame_index is out of range.")
        if not 0 <= camera_index < len(plan.camera_rig.cameras):
            raise IndexError("BLCS mesh camera_index is out of range.")
        sampled = plan.camera_rig.cameras[camera_index]
        camera = sampled.scene_camera
        if background.camera_id != camera.camera_id:
            raise ValueError("BLCS mesh background belongs to a different camera.")
        if (background.width, background.height) != (camera.width, camera.height):
            raise ValueError("BLCS mesh background resolution differs from its camera.")
        candidate_pixels: list[NDArray[np.int64]] = []
        candidate_depths: list[NDArray[np.float32]] = []
        candidate_colors: list[NDArray[np.float32]] = []
        candidate_instances: list[NDArray[np.int32]] = []
        object_index = {
            item.object_id: index for index, item in enumerate(plan.composition.objects)
        }
        camera_from_scene = camera.camera_to_scene.inverse().matrix()
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        frame = plan.composition.frames[source_frame_index]
        for instance in frame.instances:
            index = object_index[instance.object_id]
            if not bool(plan.source.present[source_frame_index, index]):
                raise ValueError(
                    "BLCS mesh composition contains an absent ball instance."
                )
            scene_from_asset = instance.scene_from_asset.rigid.matrix()
            scene_from_asset[:3, :3] *= instance.scene_from_asset.scale
            camera_from_asset = camera_from_scene @ scene_from_asset
            pixels, depths, colors = self._render_instance(
                camera_from_asset=camera_from_asset,
                intrinsics=intrinsics,
                width=camera.width,
                height=camera.height,
            )
            if not len(pixels):
                continue
            background_depth = background.depth.reshape(-1)[pixels]
            in_front_of_background = (background_depth <= 0.0) | (
                depths < background_depth
            )
            if not np.any(in_front_of_background):
                continue
            count = int(np.count_nonzero(in_front_of_background))
            candidate_pixels.append(pixels[in_front_of_background])
            candidate_depths.append(depths[in_front_of_background])
            candidate_colors.append(colors[in_front_of_background])
            candidate_instances.append(np.full(count, index + 1, dtype=np.int32))
        if not candidate_pixels:
            return ForegroundDelta(
                key=RenderSampleKey(source_frame_index, camera.camera_id),
                pixel_indices=np.empty(0, dtype=np.int32),
                rgb=np.empty((0, 3), dtype=np.float32),
                alpha=np.empty(0, dtype=np.float32),
                depth=np.empty(0, dtype=np.float32),
                instance_ids=np.empty(0, dtype=np.int32),
            )
        pixels = np.concatenate(candidate_pixels)
        depths = np.concatenate(candidate_depths)
        colors = np.concatenate(candidate_colors)
        instances = np.concatenate(candidate_instances)
        order = np.lexsort((depths, pixels))
        sorted_pixels = pixels[order]
        keep: NDArray[np.bool_] = np.ones(len(order), dtype=np.bool_)
        keep[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
        selected = order[keep]
        selected_pixels = pixels[selected].astype(np.int32, copy=False)
        return ForegroundDelta(
            key=RenderSampleKey(source_frame_index, camera.camera_id),
            pixel_indices=selected_pixels,
            rgb=colors[selected].astype(np.float32, copy=False),
            alpha=np.ones(len(selected_pixels), dtype=np.float32),
            depth=depths[selected].astype(np.float32, copy=False),
            instance_ids=instances[selected].astype(np.int32, copy=False),
        )

    def _render_instance(
        self,
        *,
        camera_from_asset: NDArray[np.float64],
        intrinsics: NDArray[np.float64],
        width: int,
        height: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
        device = self._vertices.device
        transform = torch.as_tensor(
            camera_from_asset, device=device, dtype=torch.float32
        )
        linear = transform[:3, :3]
        translation = transform[:3, 3]
        vertices = self._vertices @ linear.T + translation
        if bool(torch.all(vertices[:, 2] <= 1.0e-5)):
            return _empty_instance()
        if bool(torch.any(vertices[:, 2] <= 1.0e-5)):
            raise ValueError("BLCS mesh intersects the unsupported camera near plane.")
        normals = torch.nn.functional.normalize(self._normals @ linear.T, dim=1)
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        projected_u = fx * vertices[:, 0] / vertices[:, 2] + cx
        projected_v = fy * vertices[:, 1] / vertices[:, 2] + cy
        minimum_x = max(0, int(torch.floor(projected_u.min()).item()) - 1)
        maximum_x = min(width - 1, int(torch.ceil(projected_u.max()).item()) + 1)
        minimum_y = max(0, int(torch.floor(projected_v.min()).item()) - 1)
        maximum_y = min(height - 1, int(torch.ceil(projected_v.max()).item()) + 1)
        if minimum_x > maximum_x or minimum_y > maximum_y:
            return _empty_instance()
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(minimum_y, maximum_y + 1, device=device, dtype=torch.int64),
            torch.arange(minimum_x, maximum_x + 1, device=device, dtype=torch.int64),
            indexing="ij",
        )
        flat_x = pixel_x.reshape(-1)
        flat_y = pixel_y.reshape(-1)
        rays = torch.stack(
            (
                (flat_x.to(torch.float32) + 0.5 - cx) / fx,
                (flat_y.to(torch.float32) + 0.5 - cy) / fy,
                torch.ones_like(flat_x, dtype=torch.float32),
            ),
            dim=1,
        )
        depths, face_indices, barycentric = _ray_triangle_intersections(
            rays,
            vertices,
            self._faces,
        )
        hit = torch.isfinite(depths)
        if not bool(hit.any()):
            return _empty_instance()
        selected_faces = self._faces[face_indices[hit]]
        selected_barycentric = barycentric[hit]
        colors = torch.sum(
            self._colors[selected_faces] * selected_barycentric[:, :, None],
            dim=1,
        )
        interpolated_normals = torch.nn.functional.normalize(
            torch.sum(
                normals[selected_faces] * selected_barycentric[:, :, None], dim=1
            ),
            dim=1,
        )
        view_to_camera = -torch.nn.functional.normalize(rays[hit], dim=1)
        lambert = torch.abs(torch.sum(interpolated_normals * view_to_camera, dim=1))
        intensity = self.ambient + self.diffuse * lambert
        colors = torch.clamp(colors * intensity[:, None], 0.0, 1.0)
        pixels = (flat_y[hit] * width + flat_x[hit]).to(torch.int64)
        return (
            pixels.cpu().numpy().astype(np.int64, copy=False),
            depths[hit].cpu().numpy().astype(np.float32, copy=False),
            colors.cpu().numpy().astype(np.float32, copy=False),
        )


def _ray_triangle_intersections(
    rays: Tensor,
    vertices: Tensor,
    faces: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return closest double-sided Moller-Trumbore hits for camera-origin rays."""
    pixel_count = len(rays)
    device = rays.device
    closest = torch.full((pixel_count,), torch.inf, device=device, dtype=torch.float32)
    closest_face = torch.zeros(pixel_count, device=device, dtype=torch.int64)
    closest_barycentric = torch.zeros(
        (pixel_count, 3), device=device, dtype=torch.float32
    )
    epsilon = 1.0e-8
    for start in range(0, len(faces), _RAY_TRIANGLE_CHUNK):
        stop = min(start + _RAY_TRIANGLE_CHUNK, len(faces))
        triangles = vertices[faces[start:stop]]
        vertex_zero = triangles[:, 0]
        edge_one = triangles[:, 1] - vertex_zero
        edge_two = triangles[:, 2] - vertex_zero
        ray_cross_edge_two = torch.linalg.cross(
            rays[:, None, :], edge_two[None, :, :], dim=2
        )
        determinant = torch.sum(edge_one[None, :, :] * ray_cross_edge_two, dim=2)
        nonparallel = torch.abs(determinant) > epsilon
        inverse = torch.where(nonparallel, 1.0 / determinant, 0.0)
        origin_to_vertex = -vertex_zero
        barycentric_one = (
            torch.sum(origin_to_vertex[None, :, :] * ray_cross_edge_two, dim=2)
            * inverse
        )
        origin_cross_edge_one = torch.linalg.cross(
            origin_to_vertex,
            edge_one,
            dim=1,
        )
        barycentric_two = (
            torch.sum(rays[:, None, :] * origin_cross_edge_one[None, :, :], dim=2)
            * inverse
        )
        depth = torch.sum(edge_two * origin_cross_edge_one, dim=1)[None, :] * inverse
        valid = (
            nonparallel
            & (barycentric_one >= -epsilon)
            & (barycentric_two >= -epsilon)
            & (barycentric_one + barycentric_two <= 1.0 + epsilon)
            & (depth > 1.0e-5)
        )
        chunk_depth, local_face = torch.min(
            torch.where(valid, depth, torch.inf),
            dim=1,
        )
        replace_hit = chunk_depth < closest
        if bool(replace_hit.any()):
            row = torch.arange(pixel_count, device=device)
            selected_one = barycentric_one[row, local_face]
            selected_two = barycentric_two[row, local_face]
            selected_barycentric = torch.stack(
                (1.0 - selected_one - selected_two, selected_one, selected_two),
                dim=1,
            )
            closest = torch.where(replace_hit, chunk_depth, closest)
            closest_face = torch.where(replace_hit, local_face + start, closest_face)
            closest_barycentric = torch.where(
                replace_hit[:, None],
                selected_barycentric,
                closest_barycentric,
            )
    return closest, closest_face, closest_barycentric


def _empty_instance() -> tuple[
    NDArray[np.int64],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
    )


@dataclass(frozen=True, slots=True)
class BLCSMeshNHTRenderer:
    """Render one static NHT 3DGS background and depth-compose moving GLB balls."""

    assets: BLCSCompositionAssets
    client: NHTRenderClient
    executable: str | Path
    environment: Mapping[str, str]
    timeout_seconds: float
    execution_device: str
    maximum_batch_frames: int
    _rasterizer: BLCSMeshRasterizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.assets.rendering is not BLCSBallRendering.MESH:
            raise ValueError("BLCSMeshNHTRenderer requires assets.rendering=mesh.")
        if not isinstance(self.client, NHTRenderClient):
            raise TypeError("BLCS mesh renderer requires NHTRenderClient.")
        if self.maximum_batch_frames != 1:
            raise ValueError("BLCS mesh rendering rasterizes one frame at a time.")
        mesh = load_ball_mesh_asset(self.assets)
        object.__setattr__(
            self,
            "_rasterizer",
            BLCSMeshRasterizer(mesh=mesh, device=self.execution_device),
        )

    def validate_asset(self) -> None:
        """Signal that construction already loaded and validated the exact GLB."""
        mesh = self.assets.mesh
        if mesh is None:
            raise ValueError("BLCS mesh renderer lost its configured mesh asset.")
        if self._rasterizer.mesh.source_path != mesh.path:
            raise ValueError("BLCS mesh rasterizer source changed after construction.")

    def render(
        self,
        *,
        plans: Sequence[BLCSTrajectoryPlan],
        scene_path: Path,
        samples_directory: Path,
        metric_adapter: MetricSceneAdapter,
        attempt_token: str,
    ) -> BLCSRenderAttempt:
        """Render a complete attempt using standard NHT backgrounds plus mesh depth."""
        plan_tuple = tuple(plans)
        if not plan_tuple:
            raise ValueError(
                "BLCS mesh rendering requires at least one trajectory plan."
            )
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
        rendered: list[BLCSRenderedTrajectory] = []
        generated_bytes = 0
        cuda_peak_bytes = 0
        for plan in plan_tuple:
            trajectory, generated, peak = self._render_trajectory(
                plan=plan,
                scene_path=scene_path,
                samples_directory=samples_directory,
                metric_adapter=metric_adapter,
                session=session,
            )
            rendered.append(trajectory)
            generated_bytes += generated
            cuda_peak_bytes = max(cuda_peak_bytes, peak)
        expected_misses = sum(len(plan.camera_rig.cameras) for plan in plan_tuple)
        if session.nht_invocations != len(plan_tuple):
            raise ValueError(
                "BLCS mesh mode must invoke NHT exactly once per trajectory."
            )
        if session.background_cache_misses != expected_misses:
            raise ValueError(
                "BLCS mesh mode must load one background per trajectory-camera."
            )
        return BLCSRenderAttempt(
            attempt_token=attempt_token,
            trajectories=tuple(rendered),
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
        working_render = trajectory_directory / "nht-background-working"
        with ExitStack() as temporary:
            temporary.callback(_remove_temporary_directory, working_render)
            temporary.callback(request_path.unlink, missing_ok=True)
            command = NHTRenderCommandRequest(
                scene_path=scene_path,
                output_directory=working_render,
                arbitrary_cameras=request,
                arbitrary_request_path=request_path,
                executable=self.executable,
            )
            session.note_nht_invocation()
            background_result = self.client.render(
                command,
                environment=dict(self.environment),
                timeout_seconds=self.timeout_seconds,
            )
            if background_result.scene_id != plan.dataset_scene_id:
                raise ValueError(
                    "NHT background scene_id disagrees with the BLCS plan."
                )
            nht_generated_bytes = (
                directory_size_bytes(working_render) + request_path.stat().st_size
            )
            camera_ids = tuple(
                sampled.scene_camera.camera_id for sampled in plan.camera_rig.cameras
            )
            background_directory = trajectory_directory / "backgrounds"
            session.create_background_store(
                plan.source.trajectory_id,
                background_directory,
                rendered=background_result,
                nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
                expected_camera_ids=camera_ids,
            )
            first_camera = plan.camera_rig.cameras[0].scene_camera
            if any(
                sampled.scene_camera.width != first_camera.width
                or sampled.scene_camera.height != first_camera.height
                for sampled in plan.camera_rig.cameras
            ):
                raise ValueError("BLCS mesh chunks require one camera shape per rig.")
            writer = ChunkWriter(
                trajectory_directory / "chunks",
                attempt_token=session.attempt_token,
                camera_ids=camera_ids,
                width=first_camera.width,
                height=first_camera.height,
            )
            if self.execution_device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(self.execution_device))
            readers: list[ChunkReader] = []
            rendered_visible_object_views = 0
            for chunk in plan.chunks:
                deltas: list[ForegroundDelta] = []
                metadata: list[Mapping[str, object]] = []
                for frame_index in chunk.frame_indices:
                    for camera_index, camera_id in enumerate(camera_ids):
                        delta = self._rasterizer.render_sample(
                            plan=plan,
                            source_frame_index=frame_index,
                            camera_index=camera_index,
                            background=session.background(
                                plan.source.trajectory_id,
                                camera_id,
                            ),
                        )
                        deltas.append(delta)
                        rendered_visible_object_views += len(
                            delta.visible_instance_counts
                        )
                        metadata.append(
                            build_blcs_sample_metadata(
                                plan=plan,
                                source_frame_index=frame_index,
                                camera_index=camera_index,
                                chunk_index=chunk.chunk_index,
                                delta=delta,
                            )
                        )
                readers.append(
                    writer.write(
                        ForegroundDeltaBatch(
                            chunk_id=f"chunk-{chunk.chunk_index:06d}",
                            deltas=tuple(deltas),
                            metadata=tuple(metadata),
                        )
                    )
                )
            cuda_peak = (
                int(
                    torch.cuda.max_memory_allocated(torch.device(self.execution_device))
                )
                if self.execution_device.startswith("cuda")
                else 0
            )
            canonical_bytes = directory_size_bytes(
                background_directory
            ) + directory_size_bytes(trajectory_directory / "chunks")
            return (
                BLCSRenderedTrajectory(
                    trajectory_id=plan.source.trajectory_id,
                    directory=trajectory_directory,
                    background_directory=background_directory,
                    chunk_readers=tuple(readers),
                    rendered_visible_object_views=rendered_visible_object_views,
                ),
                nht_generated_bytes + canonical_bytes,
                cuda_peak,
            )


def _remove_temporary_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


__all__ = ["BLCSMeshNHTRenderer", "BLCSMeshRasterizer"]
