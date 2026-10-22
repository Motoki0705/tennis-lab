"""PlacementTrace must expose the production object transform bit-for-bit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tennis_scene.motion_alignment.mesh_placement import (
    CanonicalBody,
    canonicalize_incam_body,
    interpolate_rotations,
    place_canonical_body,
)
from src.tennis_scene.motion_alignment.temporal import (
    PlacementRejection,
    TemporalPlacementConfig,
    fit_supported_track,
)
from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyParameters
from src.tennis_scene.pipeline.components.gvhmr import (
    GVHMROutput,
    RecoveredBody,
    RecoveredBodySegment,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlacementTrace,
    PlayerSkeleton,
    placement_weights,
    reconstruct_player_bodies,
    trace_player_placement,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.utils.geometry.triangulation import (
    PinholeCamera,
    TriangulatedPoints,
    reject_excessive_speed,
)

FRAMES = 16
VERTICES = 12
IDENTITY = 7
REPROJECTION_PX = 20.0


class _FakeGeometry:
    """Deterministic CPU stand-in for the SMPL vertex/COCO17 reconstructors."""

    def __init__(self) -> None:
        rng = np.random.default_rng(20261017)
        self._root: NDArray[np.float64] = np.zeros(VERTICES, np.float64)
        self._root[0] = 1.0
        self._base_vertices = rng.normal(0.12, size=(VERTICES, 3)).astype(np.float32)
        self._base_vertices[0] = np.array([0.05, -0.02, 1.15], np.float32)
        self._base_joints = np.array(
            [
                [0.00, 0.00, 1.75],
                [-0.05, 0.02, 1.78],
                [0.05, 0.02, 1.78],
                [-0.10, 0.01, 1.75],
                [0.10, 0.01, 1.75],
                [-0.20, 0.00, 1.50],
                [0.20, 0.00, 1.50],
                [-0.28, 0.02, 1.25],
                [0.28, 0.02, 1.25],
                [-0.34, 0.04, 1.00],
                [0.34, 0.04, 1.00],
                [-0.12, 0.00, 1.00],
                [0.12, 0.00, 1.00],
                [-0.13, 0.00, 0.55],
                [0.13, 0.00, 0.55],
                [-0.14, 0.00, 0.10],
                [0.14, 0.00, 0.10],
            ],
            np.float32,
        )
        self._vertex_pattern = rng.normal(0.08, size=(VERTICES, 3)).astype(np.float32)
        self._vertex_pattern[0] = 0.0
        self._joint_pattern = rng.normal(0.12, size=(17, 3)).astype(np.float32)
        self._joint_pattern -= self._joint_pattern[[11, 12]].mean(0)
        self.unloaded = False
        self.calls = 0

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        return self._root

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        self.calls += 1
        pose = parameters.body_pose[:, :3]
        vertices = (
            self._base_vertices[None]
            + parameters.transl[:, None, :]
            + pose[:, None, :] * self._vertex_pattern[None]
        )
        joints = (
            self._base_joints[None]
            + 0.5 * parameters.transl[:, None, :]
            + pose[:, None, :] * self._joint_pattern[None]
        )
        return BodyGeometry(vertices.astype(np.float32), joints.astype(np.float32))

    def unload(self) -> None:
        self.unloaded = True


@dataclass(frozen=True)
class _Case:
    skeleton: PlayerSkeleton
    grouped: GroupedObservations
    raw: ObjectObservations
    cameras: tuple[PinholeCamera, ...]
    recovered: GVHMROutput
    geometry: _FakeGeometry
    config: TemporalPlacementConfig
    reference: dict[str, Any]


def _cameras() -> tuple[PinholeCamera, ...]:
    intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    first = PinholeCamera(
        "cam0",
        intrinsic,
        np.eye(3),
        np.array([0.0, 0.0, 8.0]),
    )
    second = PinholeCamera(
        "cam1",
        intrinsic,
        Rotation.from_euler("y", 0.4).as_matrix(),
        np.array([0.5, 0.0, 8.0]),
    )
    return first, second


def _segments() -> tuple[RecoveredBodySegment, ...]:
    body_pose: NDArray[np.float32] = np.zeros((FRAMES, 63), np.float32)
    body_pose[:, 0] = np.linspace(-0.25, 0.25, FRAMES, dtype=np.float32)
    orient: NDArray[np.float32] = np.zeros((FRAMES, 3), np.float32)
    orient[:, 2] = np.linspace(-0.35, 0.40, FRAMES, dtype=np.float32)
    betas: NDArray[np.float32] = np.zeros((FRAMES, 10), np.float32)
    betas[:, 0] = 0.3
    transl: NDArray[np.float32] = np.zeros((FRAMES, 3), np.float32)
    transl[:, 0] = 0.20 * np.sin(np.arange(FRAMES) * 0.7)
    transl[:, 1] = -0.15 * np.cos(np.arange(FRAMES) * 0.5)
    transl[:, 2] = 0.04 * np.arange(FRAMES)
    arrays = {
        "body_pose": body_pose,
        "global_orient": orient,
        "betas": betas,
        "transl": transl,
    }
    sources: tuple[NDArray[np.int64], ...] = (
        np.arange(0, 6, dtype=np.int64),
        np.arange(8, 14, dtype=np.int64),
    )
    return tuple(
        RecoveredBodySegment(
            source,
            BodyParameters(
                body_pose=arrays["body_pose"][source],
                global_orient=arrays["global_orient"][source],
                betas=arrays["betas"][source],
                transl=arrays["transl"][source],
            ),
            len(source),
        )
        for source in sources
    )


def _valid_mask() -> NDArray[np.bool_]:
    valid: NDArray[np.bool_] = np.zeros((FRAMES, 17), bool)
    valid[0:3] = True
    valid[3, :3] = True
    valid[4:6] = True
    valid[8:10] = True
    valid[10, :3] = True
    valid[11:14] = True
    return valid


def _dense_parameters(
    params: BodyParameters,
    source: np.ndarray,
    selected: np.ndarray,
    betas: np.ndarray,
) -> BodyParameters:
    return BodyParameters(
        interpolate_rotations(params.body_pose, source, selected),
        interpolate_rotations(params.global_orient, source, selected),
        np.repeat(betas[None], len(selected), axis=0),
        np.stack(
            [
                np.interp(selected, source, params.transl[:, column])
                for column in range(3)
            ],
            -1,
        ).astype(np.float32),
    )


def _canonical_arrays(
    geometry: _FakeGeometry,
    camera: PinholeCamera,
    segments: tuple[RecoveredBodySegment, ...],
    betas: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Canonicalize every eligible frame exactly as production does."""
    canonical_vertices: NDArray[np.float32] = np.zeros(
        (FRAMES, VERTICES, 3), np.float32
    )
    joints: NDArray[np.float64] = np.zeros((FRAMES, 17, 3), np.float64)
    offsets: NDArray[np.float64] = np.zeros((FRAMES, 3), np.float64)
    rotations: NDArray[np.float64] = np.zeros((FRAMES, 3, 3), np.float64)
    segment_ids: NDArray[np.int64] = np.full(FRAMES, -1, np.int64)
    poses: NDArray[np.float32] = np.zeros((FRAMES, 63), np.float32)
    frame_joints: NDArray[np.float64] = np.zeros((FRAMES, 17, 3), np.float64)
    for segment_id, segment in enumerate(segments):
        source = segment.source_frames
        dense: NDArray[np.int64] = np.arange(
            int(source[0]), int(source[-1]) + 1, dtype=np.int64
        )
        segment_ids[dense] = segment_id
        eligible = valid[dense].sum(-1) >= TemporalPlacementConfig().min_joints
        eligible_frames = dense[eligible]
        dense_params = _dense_parameters(segment.parameters, source, source, betas)
        geometry_values = geometry.reconstruct(dense_params)
        canonical = canonicalize_incam_body(
            geometry_values.vertices,
            geometry_values.coco17,
            dense_params.global_orient,
            camera,
            geometry.root_regressor,
        )
        frame_joints[source] = canonical.joints
        canonical_vertices[eligible_frames] = canonical.vertices[eligible]
        joints[eligible_frames] = canonical.joints[eligible]
        offsets[eligible_frames] = canonical.root_from_hip[eligible]
        rotations[eligible_frames] = canonical.rotation[eligible]
        poses[eligible_frames] = dense_params.body_pose[eligible]
    return {
        "vertices": canonical_vertices,
        "joints": joints,
        "offsets": offsets,
        "rotations": rotations,
        "segment_ids": segment_ids,
        "poses": poses,
        "frame_joints": frame_joints,
    }


def _reference_reconstruction(
    geometry: _FakeGeometry,
    cameras: tuple[PinholeCamera, ...],
    skeleton: PlayerSkeleton,
    grouped: GroupedObservations,
    segments: tuple[RecoveredBodySegment, ...],
    fps: float,
    config: TemporalPlacementConfig,
) -> dict[str, Any]:
    """The direct component pipeline: canonicalize -> fit -> place -> speed gate."""
    betas = np.average(
        np.stack([segment.parameters.betas.mean(0) for segment in segments]),
        axis=0,
        weights=[segment.observed_samples for segment in segments],
    ).astype(np.float32)
    canonical = _canonical_arrays(
        geometry, cameras[0], segments, betas, skeleton.valid[0]
    )
    weights = placement_weights(skeleton, grouped, 0, config)
    fitted = fit_supported_track(
        canonical["joints"],
        skeleton.positions[0].astype(np.float64),
        weights,
        canonical["segment_ids"],
        fps=fps,
        config=config,
    )
    position: NDArray[np.float32] = np.zeros((FRAMES, 3), np.float32)
    yaw: NDArray[np.float32] = np.zeros(FRAMES, np.float32)
    root_valid: NDArray[np.bool_] = np.zeros(FRAMES, bool)
    heading: NDArray[np.bool_] = np.zeros(FRAMES, bool)
    mesh: NDArray[np.bool_] = np.zeros(FRAMES, bool)
    orient: NDArray[np.float32] = np.zeros((FRAMES, 3), np.float32)
    vertices: NDArray[np.float32] = np.zeros((FRAMES, VERTICES, 3), np.float32)
    mesh_error: NDArray[np.float32] = np.zeros(FRAMES, np.float32)
    roots_reason = fitted.reasons.copy()
    valid_frames = np.flatnonzero(fitted.valid)
    if fitted.scale is not None and len(valid_frames):
        placed = place_canonical_body(
            CanonicalBody(
                canonical["vertices"][valid_frames],
                canonical["joints"][valid_frames],
                canonical["offsets"][valid_frames],
                canonical["rotations"][valid_frames],
            ),
            fitted.hip_position[valid_frames],
            fitted.yaw_correction[valid_frames],
            fitted.scale,
        )
        errors: NDArray[np.float64] = np.zeros(len(valid_frames), np.float64)
        for camera_index, camera in enumerate(cameras):
            projected, front = camera.project(placed.joints_court)
            visible = skeleton.inliers[0, camera_index, valid_frames].copy()
            torso = np.zeros_like(visible)
            torso[:, [5, 6, 11, 12]] = True
            visible &= torso
            residual = np.linalg.norm(
                projected - grouped.uv_px[0, camera_index, valid_frames], axis=-1
            )
            error = np.where(visible, np.where(front, residual, np.inf), 0).max(-1)
            errors = np.maximum(errors, error)
        position[valid_frames] = placed.position
        root_valid[valid_frames] = True
        roots_reason[valid_frames] = 0
        heading[valid_frames] = placed.heading_valid
        yaw[valid_frames] = placed.yaw
        mesh_error[valid_frames] = np.where(
            np.isfinite(errors), errors, np.finfo(np.float32).max
        )
        accepted = placed.heading_valid & (errors <= REPROJECTION_PX)
        mesh[valid_frames] = accepted
        orient[valid_frames[accepted]] = placed.global_orient[accepted]
        vertices[valid_frames[accepted]] = placed.vertices_local[accepted]
    gated = reject_excessive_speed(
        TriangulatedPoints(
            position,
            root_valid,
            roots_reason,
            np.zeros((len(cameras), FRAMES), bool),
            np.zeros((len(cameras), FRAMES), np.float32),
        ),
        fps=fps,
        max_speed_mps=12.0,
    )
    heading &= gated.valid
    mesh &= gated.valid
    yaw[~heading] = 0
    poses = canonical["poses"].copy()
    poses[~mesh] = 0
    orient[~mesh] = 0
    vertices[~mesh] = 0
    return {
        "betas": betas,
        "fitted": fitted,
        "weights": weights,
        "canonical": canonical,
        "position": gated.positions,
        "root_valid": gated.valid,
        "roots_reason": gated.reasons,
        "yaw": yaw,
        "heading_valid": heading,
        "mesh_valid": mesh,
        "orient": orient,
        "vertices": vertices,
        "poses": poses,
        "mesh_error": mesh_error,
    }


def _build_case() -> _Case:
    geometry = _FakeGeometry()
    cameras = _cameras()
    segments = _segments()
    valid = _valid_mask()
    betas = np.average(
        np.stack([segment.parameters.betas.mean(0) for segment in segments]),
        axis=0,
        weights=[segment.observed_samples for segment in segments],
    ).astype(np.float32)
    canonical = _canonical_arrays(geometry, cameras[0], segments, betas, valid)
    true_scale = 1.12
    target: NDArray[np.float32] = np.zeros((FRAMES, 17, 3), np.float32)
    for frame in range(6):
        angle = 0.35 + 0.06 * frame
        root = np.array([1.2 + 0.05 * frame, -1.5 + 0.08 * frame, 0.8])
        correction = Rotation.from_euler("z", angle).as_matrix()
        target[frame] = (
            true_scale * (correction @ canonical["frame_joints"][frame].T).T + root
        )
    # The second GVHMR interval is geometrically degenerate: all valid
    # COCO17 targets collapse to one point, so yaw cannot be observed.
    target[8:14] = np.array([2.0, -0.5, 1.0], np.float32)
    target[~valid] = 0

    cameras_tuple = cameras
    uv: NDArray[np.float32] = np.zeros((1, 2, FRAMES, 17, 2), np.float32)
    for camera_index, camera in enumerate(cameras_tuple):
        projected, front = camera.project(target.astype(np.float64))
        assert front[valid].all()
        uv[0, camera_index, valid] = projected[valid].astype(np.float32)
    confidence = np.where(valid, 0.9, 0.0)[None, None].astype(np.float32)
    confidence = np.broadcast_to(confidence, (1, 2, FRAMES, 17)).copy()
    inliers = np.broadcast_to(valid[None, None], (1, 2, FRAMES, 17)).copy()
    skeleton = PlayerSkeleton(
        target[None].copy(),
        valid[None].copy(),
        np.zeros((1, FRAMES, 17), np.uint8),
        inliers,
        np.where(inliers, 2.0, 0.0).astype(np.float32),
    )
    grouped = GroupedObservations(
        np.array([IDENTITY], np.int64),
        uv,
        confidence,
        np.broadcast_to(valid[None, None], (1, 2, FRAMES, 17)).copy(),
        np.zeros((1, 2, FRAMES), np.int64),
    )
    raw = ObjectObservations(
        ("cam0", "cam1"),
        (1280, 720),
        30.0,
        np.zeros((2, FRAMES, 1, 17, 2), np.float32),
        np.zeros((2, FRAMES, 1, 17), np.float32),
        np.ones((2, FRAMES, 1), bool),
        np.zeros((2, 1), np.int64),
        np.zeros((2, FRAMES, 1, 3), np.float32),
    )
    recovered = GVHMROutput((RecoveredBody(IDENTITY, "cam0", _segments()),))
    config = TemporalPlacementConfig()
    reference = _reference_reconstruction(
        geometry,
        cameras_tuple,
        skeleton,
        grouped,
        segments,
        raw.fps,
        config,
    )
    return _Case(
        skeleton,
        grouped,
        raw,
        cameras_tuple,
        recovered,
        geometry,
        config,
        reference,
    )


def _reconstruct(case: _Case) -> Any:
    return reconstruct_player_bodies(
        case.grouped,
        case.raw,
        case.skeleton,
        case.cameras,
        case.recovered,
        body=case.geometry,
        reprojection_px=REPROJECTION_PX,
        placement_config=case.config,
    )


def _trace(case: _Case) -> PlacementTrace:
    return trace_player_placement(
        case.recovered.bodies[0],
        case.skeleton,
        case.grouped,
        case.cameras,
        player=0,
        body=case.geometry,
        fps=case.raw.fps,
        placement_config=case.config,
    )


def test_trace_fields_shapes_and_segment_cameras() -> None:
    case = _build_case()
    trace = _trace(case)
    frames = np.r_[np.arange(0, 6), np.arange(8, 14)]
    assert trace.identity == IDENTITY
    np.testing.assert_array_equal(trace.frames, frames)
    np.testing.assert_array_equal(
        trace.segment_ids, np.r_[np.zeros(6, np.int64), np.ones(6, np.int64)]
    )
    assert trace.segment_camera_ids == ("cam0", "cam0")
    assert trace.frames.dtype == np.int64
    assert trace.segment_ids.dtype == np.int64
    assert trace.vertices_local.dtype == np.float32
    assert trace.joints_local.dtype == np.float64
    assert trace.target_joints.dtype == np.float32
    assert trace.target_weights.dtype == np.float64
    assert trace.hip_position.dtype == np.float64
    assert trace.yaw_correction.dtype == np.float64
    assert trace.valid.dtype == np.bool_
    assert trace.reasons.dtype == np.uint8
    assert trace.frames.shape == (12,)
    assert trace.vertices_local.shape == (12, VERTICES, 3)
    assert trace.joints_local.shape == (12, 17, 3)
    assert trace.target_joints.shape == (12, 17, 3)
    assert trace.target_weights.shape == (12, 17)
    assert trace.hip_position.shape == (12, 3)
    assert trace.yaw_correction.shape == (12,)


def test_trace_object_transform_matches_place_canonical_body() -> None:
    case = _build_case()
    trace = _trace(case)
    canonical = case.reference["canonical"]
    body_mask = case.skeleton.valid[0].sum(-1) >= case.config.min_joints
    body_frames = np.flatnonzero(body_mask)
    assert trace.scale is not None
    for frame in body_frames:
        index = int(np.flatnonzero(trace.frames == frame)[0])
        placed = place_canonical_body(
            CanonicalBody(
                canonical["vertices"][frame : frame + 1],
                canonical["joints"][frame : frame + 1],
                canonical["offsets"][frame : frame + 1],
                canonical["rotations"][frame : frame + 1],
            ),
            trace.hip_position[index : index + 1],
            trace.yaw_correction[index : index + 1],
            trace.scale,
        )
        correction = Rotation.from_euler("z", trace.yaw_correction[index]).as_matrix()
        world = (
            trace.hip_position[index]
            + trace.scale
            * (correction @ trace.vertices_local[index].astype(np.float64).T).T
        )
        np.testing.assert_allclose(
            world,
            placed.vertices_court[0],
            atol=1e-5,
            err_msg=f"frame {frame}",
        )
    np.testing.assert_array_equal(
        trace.joints_local,
        canonical["joints"][trace.frames],
    )
    expected_u = (
        np.einsum(
            "tij,tvj->tvi",
            canonical["rotations"][body_frames],
            canonical["vertices"][body_frames].astype(np.float64),
        )
        + canonical["offsets"][body_frames, None, :]
    )
    body_indices = np.searchsorted(trace.frames, body_frames)
    np.testing.assert_allclose(
        trace.vertices_local[body_indices],
        expected_u.astype(np.float32),
        atol=1e-6,
    )


def test_trace_validity_and_reconstruction_match_reference() -> None:
    case = _build_case()
    trace = _trace(case)
    reference = case.reference
    players = _reconstruct(case)
    np.testing.assert_array_equal(players.root_valid[0, trace.frames], trace.valid)
    np.testing.assert_array_equal(players.root_reasons[0, trace.frames], trace.reasons)
    np.testing.assert_array_equal(players.root_valid[0], reference["root_valid"])
    np.testing.assert_array_equal(players.root_reasons[0], reference["roots_reason"])
    np.testing.assert_array_equal(players.position[0], reference["position"])
    np.testing.assert_array_equal(players.heading_valid[0], reference["heading_valid"])
    np.testing.assert_array_equal(players.smpl_valid[0], reference["mesh_valid"])
    np.testing.assert_array_equal(players.betas[0], reference["betas"])
    np.testing.assert_array_equal(players.body_pose[0], reference["poses"])
    np.testing.assert_array_equal(players.vertices_local[0], reference["vertices"])
    np.testing.assert_array_equal(players.global_orient[0], reference["orient"])
    np.testing.assert_array_equal(players.yaw[0], reference["yaw"])
    np.testing.assert_array_equal(
        players.mesh_reprojection_px[0], reference["mesh_error"]
    )
    assert trace.scale == reference["fitted"].scale
    np.testing.assert_array_equal(
        trace.hip_position, reference["fitted"].hip_position[trace.frames]
    )
    np.testing.assert_array_equal(
        trace.yaw_correction,
        reference["fitted"].yaw_correction[trace.frames],
    )
    np.testing.assert_array_equal(
        trace.target_weights, reference["weights"][trace.frames]
    )
    assert players.diagnostics["players"][str(IDENTITY)]["scale"] == trace.scale
    assert players.diagnostics["source_cameras"][str(IDENTITY)] == "cam0"


def test_trace_reports_missing_and_rejected_frames() -> None:
    case = _build_case()
    trace = _trace(case)
    by_frame = {int(frame): index for index, frame in enumerate(trace.frames)}
    assert trace.reasons[by_frame[0]] == 0
    assert trace.valid[by_frame[0]]
    assert trace.reasons[by_frame[3]] == PlacementRejection.INSUFFICIENT_JOINTS
    assert not trace.valid[by_frame[3]]
    assert trace.reasons[by_frame[8]] == PlacementRejection.UNOBSERVABLE_YAW
    assert not trace.valid[by_frame[8]]
    for frame in (3, 10):
        np.testing.assert_array_equal(
            trace.vertices_local[by_frame[frame]], np.zeros((VERTICES, 3))
        )
    outside = np.setdiff1d(np.arange(FRAMES), trace.frames)
    players = _reconstruct(case)
    np.testing.assert_array_equal(
        players.root_reasons[0, outside],
        np.full(len(outside), 1, np.uint8),
    )


def test_trace_target_joints_use_nan_for_invalid_joints() -> None:
    case = _build_case()
    trace = _trace(case)
    expected_valid = case.skeleton.valid[0, trace.frames]
    assert not expected_valid.all()
    nan_mask = ~expected_valid
    assert np.isnan(trace.target_joints[nan_mask]).all()
    assert np.isfinite(trace.target_joints[expected_valid]).all()
    np.testing.assert_array_equal(
        trace.target_joints[expected_valid],
        case.skeleton.positions[0, trace.frames][expected_valid],
    )
    assert (trace.target_weights[nan_mask] == 0).all()
    assert (trace.target_weights > 0)[expected_valid].any()


def test_trace_does_not_unload_or_mutate_inputs() -> None:
    case = _build_case()
    positions = case.skeleton.positions.copy()
    inliers = case.skeleton.inliers.copy()
    uv = case.grouped.uv_px.copy()
    trace = _trace(case)
    assert not case.geometry.unloaded
    np.testing.assert_array_equal(case.skeleton.positions, positions)
    np.testing.assert_array_equal(case.skeleton.inliers, inliers)
    np.testing.assert_array_equal(case.grouped.uv_px, uv)
    assert trace.valid.any()


def test_speed_rejection_is_visible_and_matches_reconstruction() -> None:
    case = _build_case()
    case.skeleton.positions[0, 2] += np.array([20.0, 0.0, 0.0], np.float32)
    trace = _trace(case)
    players = _reconstruct(case)
    assert (trace.reasons == 6).sum() == 3
    assert not trace.valid[trace.reasons == 6].any()
    np.testing.assert_array_equal(trace.valid, players.root_valid[0, trace.frames])
    np.testing.assert_array_equal(trace.reasons, players.root_reasons[0, trace.frames])


def test_empty_recovered_body_keeps_the_existing_missing_output() -> None:
    case = _build_case()
    empty = GVHMROutput((RecoveredBody(IDENTITY, "cam0", ()),))
    trace = trace_player_placement(
        empty.bodies[0],
        case.skeleton,
        case.grouped,
        case.cameras,
        player=0,
        body=case.geometry,
        fps=case.raw.fps,
        placement_config=case.config,
    )
    assert trace.frames.size == 0
    assert trace.vertices_local.shape == (0, 0, 3)
    assert trace.scale is None
    assert trace.segment_camera_ids == ()
    players = reconstruct_player_bodies(
        case.grouped,
        case.raw,
        case.skeleton,
        case.cameras,
        empty,
        body=case.geometry,
        reprojection_px=REPROJECTION_PX,
        placement_config=case.config,
    )
    assert players.vertices_local is None
    assert not players.root_valid.any()
    np.testing.assert_array_equal(players.root_reasons[0], np.ones(FRAMES, np.uint8))
    assert players.diagnostics["source_cameras"][str(IDENTITY)] == "cam0"
    assert str(IDENTITY) not in players.diagnostics["players"]
