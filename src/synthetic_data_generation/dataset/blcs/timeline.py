"""Deterministic court, camera, composition, and chunk planning for BLCS."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.composition import (
    GaussianDeformationKind,
    GaussianFrame,
    GaussianInstance,
    GaussianSceneComposition,
    GaussianSceneObject,
    GaussianTransform,
)
from src.synthetic_data_generation.dataset.blcs.contracts import (
    BLCSChunk,
    BLCSCompositionAssets,
    BLCSTrajectory,
)
from src.synthetic_data_generation.dataset.camera_profiles import (
    CameraProfileConfig,
    SampledCameraRig,
    assert_projection_equivalent,
    sample_camera_rig,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court_assignment import (
    assign_courts_balanced,
)
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


@dataclass(frozen=True, slots=True)
class BLCSTrajectoryPlan:
    """One complete source trajectory placed on one accepted target court."""

    dataset_scene_id: str
    source: BLCSTrajectory
    global_frame_offset: int
    target_court: TargetCourtBinding
    camera_rig: SampledCameraRig
    composition: GaussianSceneComposition
    chunks: tuple[BLCSChunk, ...]
    positions_scene: NDArray[np.float64]
    camera_uv: NDArray[np.float64]
    camera_depth: NDArray[np.float64]
    geometric_visible: NDArray[np.bool_]
    court_uv: NDArray[np.float64]
    court_visible: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.global_frame_offset < 0:
            raise ValueError("global_frame_offset must be non-negative.")
        if self.camera_rig.court_instance_id != self.target_court.court_instance_id:
            raise ValueError("BLCS camera rig and trajectory target different courts.")
        if self.composition.scene_id != self.dataset_scene_id:
            raise ValueError("BLCS composition scene_id disagrees with its workspace.")
        if len(self.composition.frames) != self.source.frame_count:
            raise ValueError("BLCS composition must carry every source frame.")
        chunk_indices = tuple(
            frame_index for chunk in self.chunks for frame_index in chunk.frame_indices
        )
        if chunk_indices != tuple(range(self.source.frame_count)):
            raise ValueError(
                "BLCS chunks must cover source frames exactly once in order."
            )
        positions = _finite_array(self.positions_scene, name="positions_scene")
        uv = _finite_array(self.camera_uv, name="camera_uv")
        depth = _finite_array(self.camera_depth, name="camera_depth")
        visible = np.asarray(self.geometric_visible)
        court_uv = _finite_array(self.court_uv, name="court_uv")
        court_visible = np.asarray(self.court_visible)
        if visible.dtype != np.bool_:
            raise TypeError("geometric_visible must use bool dtype.")
        if court_visible.dtype != np.bool_:
            raise TypeError("court_visible must use bool dtype.")
        visible = np.array(visible, dtype=np.bool_, order="C", copy=True)
        expected_position_shape = (
            self.source.frame_count,
            self.source.object_count,
            3,
        )
        camera_count = len(self.camera_rig.cameras)
        if positions.shape != expected_position_shape:
            raise ValueError("positions_scene has the wrong BLCS timeline shape.")
        if uv.shape != (
            self.source.frame_count,
            camera_count,
            self.source.object_count,
            2,
        ):
            raise ValueError("camera_uv has the wrong BLCS timeline shape.")
        if (
            depth.shape
            != (
                self.source.frame_count,
                camera_count,
                self.source.object_count,
            )
            or visible.shape != depth.shape
        ):
            raise ValueError("BLCS camera depth/visibility shape is invalid.")
        if np.any(visible & ~self.source.present[:, None, :]):
            raise ValueError("An absent BLCS object cannot be geometrically visible.")
        if np.any(visible & (depth <= 0.0)):
            raise ValueError(
                "A geometrically visible BLCS object needs positive depth."
            )
        if court_uv.shape != (camera_count, 20, 2) or court_visible.shape != (
            camera_count,
            20,
        ):
            raise ValueError("BLCS court projection has the wrong shape.")
        court_visible = np.array(court_visible, dtype=np.bool_, order="C", copy=True)
        for array in (positions, uv, depth, visible, court_uv, court_visible):
            array.setflags(write=False)
        object.__setattr__(self, "positions_scene", positions)
        object.__setattr__(self, "camera_uv", uv)
        object.__setattr__(self, "camera_depth", depth)
        object.__setattr__(self, "geometric_visible", visible)
        object.__setattr__(self, "court_uv", court_uv)
        object.__setattr__(self, "court_visible", court_visible)

    @property
    def global_frame_indices(self) -> tuple[int, ...]:
        """Return this trajectory's contiguous dataset-global frame inventory."""
        return tuple(
            range(
                self.global_frame_offset,
                self.global_frame_offset + self.source.frame_count,
            )
        )

    def to_dict(self) -> dict[str, object]:
        """Return complete semantic plan metadata without artifact identity fields."""
        return {
            "trajectory_id": self.source.trajectory_id,
            "split": self.source.split,
            "fps": self.source.fps,
            "source_frame_count": self.source.frame_count,
            "global_frame_offset": self.global_frame_offset,
            "global_frame_indices": list(self.global_frame_indices),
            "tracks": [track.to_dict() for track in self.source.tracks],
            "target_court": self.target_court.to_dict(),
            "camera_profile": self.camera_rig.profile,
            "camera_seed": self.camera_rig.seed,
            "cameras": [camera.to_metadata() for camera in self.camera_rig.cameras],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "composition": self.composition.to_dict(),
            "source_metadata": dict(self.source.source_metadata),
        }


def build_blcs_plans(
    trajectories: Sequence[BLCSTrajectory],
    *,
    dataset_scene_id: str,
    layout: MultiCourtLayout,
    camera_config: CameraProfileConfig,
    assets: BLCSCompositionAssets,
    seed: int,
    chunk_size_frames: int,
) -> tuple[BLCSTrajectoryPlan, ...]:
    """Build deterministic balanced full-source plans with one common court transform."""
    sources = tuple(trajectories)
    if not sources:
        raise ValueError("BLCS production requires at least one source trajectory.")
    if len({source.trajectory_id for source in sources}) != len(sources):
        raise ValueError("BLCS trajectory_id values must be unique.")
    if chunk_size_frames <= 0:
        raise ValueError("chunk_size_frames must be positive.")
    ordered = tuple(sorted(sources, key=lambda source: source.trajectory_id))
    assignments = assign_courts_balanced(
        {source.trajectory_id: source.split for source in ordered},
        layout=layout,
        seed=seed,
    )
    assignment_by_id = {assignment.scene_id: assignment for assignment in assignments}
    plans: list[BLCSTrajectoryPlan] = []
    global_offset = 0
    for trajectory_index, source in enumerate(ordered):
        assignment = assignment_by_id[source.trajectory_id]
        court = layout.court(assignment.court_instance_id)
        camera_seed = seed + trajectory_index
        camera_rig = sample_camera_rig(camera_config, seed=camera_seed, court=court)
        _validate_projection_authority(camera_rig=camera_rig, court=court)
        binding = TargetCourtBinding(
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            scene_from_court=court.scene_from_court,
            selection_seed=assignment.selection_seed,
        )
        positions_scene = court.scene_from_court.apply(source.positions_court_m)
        composition = _build_composition(
            dataset_scene_id=dataset_scene_id,
            source=source,
            positions_scene=positions_scene,
            court_transform=court.scene_from_court,
            assets=assets,
        )
        uv, depth, visible = _project(
            positions_scene=positions_scene,
            present=source.present,
            camera_rig=camera_rig,
        )
        court_points = (
            court_keypoints_3d(STANDARD_COURT_CONFIG).numpy().astype(np.float64)
        )
        court_scene = court.scene_from_court.apply(court_points)
        court_uv_rows = []
        court_visible_rows = []
        for sampled in camera_rig.cameras:
            pixels, court_depth = sampled.scene_camera.project_scene_points(court_scene)
            court_uv_rows.append(pixels)
            court_visible_rows.append(
                (court_depth > 0.0)
                & (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] < sampled.scene_camera.width)
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] < sampled.scene_camera.height)
            )
        chunks = tuple(
            BLCSChunk(
                chunk_index=chunk_index,
                frame_indices=tuple(
                    range(
                        start,
                        min(start + chunk_size_frames, source.frame_count),
                    )
                ),
            )
            for chunk_index, start in enumerate(
                range(0, source.frame_count, chunk_size_frames)
            )
        )
        plans.append(
            BLCSTrajectoryPlan(
                dataset_scene_id=dataset_scene_id,
                source=source,
                global_frame_offset=global_offset,
                target_court=binding,
                camera_rig=camera_rig,
                composition=composition,
                chunks=chunks,
                positions_scene=positions_scene,
                camera_uv=uv,
                camera_depth=depth,
                geometric_visible=visible,
                court_uv=np.stack(court_uv_rows),
                court_visible=np.stack(court_visible_rows),
            )
        )
        global_offset += source.frame_count
    return tuple(plans)


def _build_composition(
    *,
    dataset_scene_id: str,
    source: BLCSTrajectory,
    positions_scene: NDArray[np.float64],
    court_transform: RigidTransform,
    assets: BLCSCompositionAssets,
) -> GaussianSceneComposition:
    objects = tuple(
        GaussianSceneObject(
            object_id=track.object_id,
            instance_id=object_index + 1,
            asset_id=assets.ball.asset_id,
            deformation_kind=GaussianDeformationKind.RIGID,
        )
        for object_index, track in enumerate(source.tracks)
    )
    court_rotation = court_transform.matrix()[:3, :3]
    frames: list[GaussianFrame] = []
    for frame_index in range(source.frame_count):
        instances: list[GaussianInstance] = []
        for object_index, track in enumerate(source.tracks):
            source_frame_index = track.source_frame_indices[frame_index]
            if source_frame_index is None:
                continue
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = court_rotation
            matrix[:3, 3] = positions_scene[frame_index, object_index]
            instances.append(
                GaussianInstance(
                    object_id=track.object_id,
                    source_frame_index=source_frame_index,
                    scene_from_asset=GaussianTransform(
                        scale=1.0,
                        rigid=RigidTransform.from_matrix(matrix),
                    ),
                )
            )
        frames.append(
            GaussianFrame(frame_index=frame_index, instances=tuple(instances))
        )
    return GaussianSceneComposition(
        scene_id=dataset_scene_id,
        composition_id=f"blcs-{source.trajectory_id}",
        background=assets.background,
        assets=(assets.ball,),
        objects=objects,
        frames=tuple(frames),
    )


def _project(
    *,
    positions_scene: NDArray[np.float64],
    present: NDArray[np.bool_],
    camera_rig: SampledCameraRig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    frame_count, object_count, _ = positions_scene.shape
    camera_count = len(camera_rig.cameras)
    uv = np.zeros((frame_count, camera_count, object_count, 2), dtype=np.float64)
    depth = np.zeros((frame_count, camera_count, object_count), dtype=np.float64)
    visible = np.zeros((frame_count, camera_count, object_count), dtype=np.bool_)
    for camera_index, sampled in enumerate(camera_rig.cameras):
        camera = sampled.scene_camera
        scene_to_camera = camera.camera_to_scene.inverse()
        points_camera = scene_to_camera.apply(positions_scene)
        z = points_camera[..., 2]
        positive = z > 0.0
        safe_z = np.where(positive, z, 1.0)
        intrinsic = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        homogeneous = points_camera @ intrinsic.T
        projected = homogeneous[..., :2] / safe_z[..., None]
        in_frame = (
            (projected[..., 0] >= 0.0)
            & (projected[..., 0] < camera.width)
            & (projected[..., 1] >= 0.0)
            & (projected[..., 1] < camera.height)
        )
        uv[:, camera_index] = projected
        depth[:, camera_index] = z
        visible[:, camera_index] = present & positive & in_frame
    return uv, depth, visible


def _validate_projection_authority(
    *, camera_rig: SampledCameraRig, court: object
) -> None:
    from src.synthetic_data_generation.scene_contract import CourtInstance

    if not isinstance(court, CourtInstance):
        raise TypeError("BLCS target court must be a CourtInstance.")
    points = np.asarray(
        ((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (0.0, 1.0, 0.5)),
        dtype=np.float64,
    )
    for camera in camera_rig.cameras:
        assert_projection_equivalent(camera, court, points, atol=1.0e-6)


def _finite_array(value: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


__all__ = ["BLCSTrajectoryPlan", "build_blcs_plans"]
