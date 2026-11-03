"""Placement traces and scene bodies obey independent metric and mask contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tennis_scene.motion_alignment.temporal import (
    PlacementRejection,
    TemporalPlacementConfig,
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
    ReconstructedPlayers,
    reconstruct_player_bodies,
    trace_player_placement,
)
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.utils.geometry.triangulation import PinholeCamera

FRAMES = 16
VERTICES = 4
IDENTITY = 7
REPROJECTION_PX = 20.0
# The hip midpoint is (0, 0, 1), distinct from vertex 0 / SMPL joint0.
JOINTS = np.array(
    [
        [0.0, 0.0, 1.75],
        [-0.0625, 0.125, 1.75],
        [0.0625, 0.125, 1.75],
        [-0.125, 0.0625, 1.75],
        [0.125, 0.0625, 1.75],
        [-0.25, 0.0, 1.5],
        [0.25, 0.0, 1.5],
        [-0.375, 0.125, 1.25],
        [0.375, 0.125, 1.25],
        [-0.5, 0.25, 1.0],
        [0.5, 0.25, 1.0],
        [-0.125, 0.0, 1.0],
        [0.125, 0.0, 1.0],
        [-0.125, 0.0, 0.5],
        [0.125, 0.0, 0.5],
        [-0.125, 0.0, 0.125],
        [0.125, 0.0, 0.125],
    ],
    np.float32,
)
VERTEX_POINTS = np.array(
    [[0.25, -0.125, 1.125], [0.5, 0, 1.5], [-0.25, 0.375, 0.75], [0.125, -0.25, 1.75]],
    np.float32,
)


class _FakeGeometry:
    """Tiny articulated mesh; only the external SMPL reconstructor is replaced."""

    def __init__(self) -> None:
        self.unloaded = False
        self.calls: list[BodyParameters] = []

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        return np.array([1.0, 0.0, 0.0, 0.0])

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
        self.calls.append(parameters)
        vertices = np.broadcast_to(
            VERTEX_POINTS, (len(parameters.body_pose), VERTICES, 3)
        ).copy()
        vertices[:, 3, 0] += parameters.body_pose[:, 0]
        rotation = Rotation.from_rotvec(parameters.global_orient).as_matrix()
        vertices = (
            np.einsum("tij,tvj->tvi", rotation, vertices) + parameters.transl[:, None]
        )
        joints = np.einsum("tij,kj->tki", rotation, JOINTS) + parameters.transl[:, None]
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


def _build_case(*, camera_quarter_turn: bool = False) -> _Case:
    intrinsic = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    first_rotation = (
        np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1]])
        if camera_quarter_turn
        else np.eye(3)
    )
    cameras = (
        PinholeCamera("cam0", intrinsic, first_rotation, np.array([0.0, 0, 8])),
        PinholeCamera(
            "cam1",
            intrinsic,
            Rotation.from_euler("y", 0.4).as_matrix(),
            np.array([0.5, 0, 8.0]),
        ),
    )
    segments = []
    for source, beta in (
        (np.array([0, 5], np.int64), 0.25),
        (np.array([8, 11, 13], np.int64), 0.75),
    ):
        pose: NDArray[np.float32] = np.zeros((len(source), 63), np.float32)
        pose[:, 0] = 0.02 * source
        orient: NDArray[np.float32] = np.zeros((len(source), 3), np.float32)
        orient[:, 2] = np.pi / 2 if camera_quarter_turn else 0
        betas: NDArray[np.float32] = np.zeros((len(source), 10), np.float32)
        betas[:, 0] = beta
        # Camera-space translation cancels; it must not enter court placement.
        transl = np.column_stack(
            (2 + 0.1 * source, np.full(len(source), -0.5), np.full(len(source), 3))
        ).astype(np.float32)
        segments.append(
            RecoveredBodySegment(
                source, BodyParameters(pose, orient, betas, transl), len(source)
            )
        )

    valid: NDArray[np.bool_] = np.zeros((FRAMES, 17), bool)
    valid[:6] = True
    valid[8:14] = True
    valid[[3, 10], 3:] = False
    # A hand-specified known solution: scale 5/4, yaw +90 degrees,
    # hip (3/2, -2, 5/4). No production placement helper generates targets.
    target: NDArray[np.float32] = np.zeros((FRAMES, 17, 3), np.float32)
    target[:6, :, 0] = 1.5 - 1.25 * JOINTS[:, 1]
    target[:6, :, 1] = -2 + 1.25 * JOINTS[:, 0]
    target[:6, :, 2] = 1.25 + 1.25 * (JOINTS[:, 2] - 1)
    target[8:14] = [2, -0.5, 1]  # Collapsed bones cannot determine yaw.
    target[~valid] = 0
    uv: NDArray[np.float32] = np.zeros((1, 2, FRAMES, 17, 2), np.float32)
    for view, camera in enumerate(cameras):
        projected, front = camera.project(target.astype(np.float64))
        assert front[valid].all()
        uv[0, view, valid] = projected[valid]
    inliers = np.broadcast_to(valid[None, None], (1, 2, FRAMES, 17)).copy()
    confidence = np.where(inliers, np.array([0.8, 1.0])[None, :, None, None], 0).astype(
        np.float32
    )
    skeleton = PlayerSkeleton(
        target[None].copy(),
        valid[None].copy(),
        np.zeros((1, FRAMES, 17), np.uint8),
        inliers,
        np.where(inliers, 2, 0).astype(np.float32),
    )
    grouped = GroupedObservations(
        np.array([IDENTITY], np.int64),
        uv,
        confidence,
        inliers.copy(),
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
    return _Case(
        skeleton,
        grouped,
        raw,
        cameras,
        GVHMROutput((RecoveredBody(IDENTITY, "cam0", tuple(segments)),)),
        _FakeGeometry(),
        TemporalPlacementConfig(),
    )


def _reconstruct(case: _Case) -> ReconstructedPlayers:
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
    trace = _trace(_build_case())
    frames = np.r_[np.arange(6), np.arange(8, 14)]
    assert trace.identity == IDENTITY
    np.testing.assert_array_equal(trace.frames, frames)
    np.testing.assert_array_equal(trace.segment_ids, [0] * 6 + [1] * 6)
    assert trace.segment_camera_ids == ("cam0", "cam0")
    for value, shape, dtype in (
        (trace.frames, (12,), np.int64),
        (trace.segment_ids, (12,), np.int64),
        (trace.vertices_local, (12, VERTICES, 3), np.float32),
        (trace.joints_local, (12, 17, 3), np.float64),
        (trace.target_joints, (12, 17, 3), np.float32),
        (trace.target_weights, (12, 17), np.float64),
        (trace.hip_position, (12, 3), np.float64),
        (trace.yaw_correction, (12,), np.float64),
        (trace.valid, (12,), np.bool_),
        (trace.reasons, (12,), np.uint8),
    ):
        assert value.shape == shape
        assert value.dtype == dtype


@pytest.mark.parametrize(
    "camera_quarter_turn", [False, True], ids=["identity-camera", "rotated-camera"]
)
def test_trace_recovers_known_transform(camera_quarter_turn: bool) -> None:
    case = _build_case(camera_quarter_turn=camera_quarter_turn)
    trace = _trace(case)
    assert trace.scale == pytest.approx(1.25, abs=1e-6)
    for index, frame in enumerate(trace.frames):
        if frame in (3, 10):
            np.testing.assert_array_equal(trace.vertices_local[index], 0)
            np.testing.assert_array_equal(trace.joints_local[index], 0)
            continue
        expected_u = np.array(
            [
                [0.25, -0.125, 0.125],
                [0.5, 0, 0.5],
                [-0.25, 0.375, -0.25],
                [0.125 + 0.02 * frame, -0.25, 0.75],
            ]
        )
        np.testing.assert_allclose(trace.vertices_local[index], expected_u, atol=1e-6)
        np.testing.assert_allclose(
            trace.joints_local[index], JOINTS - [0, 0, 1], atol=1e-6
        )
        if not trace.valid[index]:
            continue
        np.testing.assert_allclose(
            trace.hip_position[index], [1.5, -2, 1.25], atol=1e-6
        )
        assert trace.yaw_correction[index] == pytest.approx(np.pi / 2, abs=1e-6)
        world = (
            trace.hip_position[index]
            + trace.scale
            * (
                Rotation.from_euler("z", trace.yaw_correction[index]).as_matrix()
                @ trace.vertices_local[index].T
            ).T
        )
        expected_world = np.array(
            [
                [1.65625, -1.6875, 1.40625],
                [1.5, -1.375, 1.875],
                [1.03125, -2.3125, 0.9375],
                [1.8125, -1.84375 + 0.025 * frame, 2.1875],
            ]
        )
        np.testing.assert_allclose(world, expected_world, atol=2e-6)


def test_reconstruction_preserves_known_body_and_renderer_transform() -> None:
    case = _build_case()
    trace = _trace(case)
    players = _reconstruct(case)
    expected_valid: NDArray[np.bool_] = np.zeros(FRAMES, bool)
    expected_valid[[0, 1, 2, 4, 5]] = True
    for mask in (
        players.root_valid[0],
        players.heading_valid[0],
        players.smpl_valid[0],
    ):
        np.testing.assert_array_equal(mask, expected_valid)
    np.testing.assert_array_equal(trace.valid, expected_valid[trace.frames])
    np.testing.assert_array_equal(players.root_reasons[0, trace.frames], trace.reasons)
    # (2 * 1/4 + 3 * 3/4) / 5 = 11/20, shared by both segments.
    np.testing.assert_allclose(players.betas[0], [0.55] + [0] * 9, atol=1e-7)
    for parameters in case.geometry.calls:
        np.testing.assert_allclose(
            parameters.betas,
            np.tile([0.55] + [0] * 9, (len(parameters.betas), 1)),
            atol=1e-7,
        )
    assert players.vertices_local is not None
    for frame in np.flatnonzero(expected_valid):
        np.testing.assert_allclose(
            players.position[0, frame], [1.65625, -1.6875, 1.40625], atol=1e-6
        )
        assert players.yaw[0, frame] == pytest.approx(np.pi / 2, abs=1e-6)
        np.testing.assert_allclose(
            players.global_orient[0, frame], [np.pi / 2, 0, 0], atol=1e-6
        )
        expected_pose: NDArray[np.float32] = np.zeros(63)
        expected_pose[0] = 0.02 * frame
        np.testing.assert_allclose(
            players.body_pose[0, frame], expected_pose, atol=1e-7
        )
        expected_vertices = 1.25 * np.array(
            [
                [0, 0, 0],
                [0.25, 0.125, 0.375],
                [-0.5, 0.5, -0.375],
                [-0.125 + 0.02 * frame, -0.125, 0.625],
            ]
        )
        np.testing.assert_allclose(
            players.vertices_local[0, frame], expected_vertices, atol=1e-6
        )
        # The renderer consumes position/yaw/global_orient/vertices_local.
        y_up_to_z_up = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        rotation = (
            Rotation.from_euler("z", players.yaw[0, frame]).as_matrix()
            @ y_up_to_z_up
            @ Rotation.from_rotvec(players.global_orient[0, frame]).as_matrix().T
        )
        world = (
            players.vertices_local[0, frame] @ rotation.T + players.position[0, frame]
        )
        expected_world = np.array(
            [
                [1.65625, -1.6875, 1.40625],
                [1.5, -1.375, 1.875],
                [1.03125, -2.3125, 0.9375],
                [1.8125, -1.84375 + 0.025 * frame, 2.1875],
            ]
        )
        np.testing.assert_allclose(world, expected_world, atol=2e-6)
    for value in (
        players.position[0],
        players.yaw[0],
        players.body_pose[0],
        players.global_orient[0],
        players.vertices_local[0],
    ):
        np.testing.assert_array_equal(value[~expected_valid], 0)
    np.testing.assert_allclose(players.mesh_reprojection_px[0], 0, atol=1e-3)
    assert players.diagnostics["players"][str(IDENTITY)]["scale"] == trace.scale
    assert players.diagnostics["source_cameras"][str(IDENTITY)] == "cam0"
    assert case.geometry.unloaded


def test_reprojection_gate_preserves_roots_and_ignores_non_torso_outliers() -> None:
    case = _build_case()
    case.grouped.uv_px[0, 1, 1, 6] += [21.0, 0]
    case.grouped.uv_px[0, 0, 2, 0] += [100.0, 0]  # Face is not a torso gate.
    case.skeleton.inliers[0, 0, 4, 6] = False
    case.grouped.uv_px[0, 0, 4, 6] += [100.0, 0]  # A rejected view is not evidence.
    trace = _trace(case)
    players = _reconstruct(case)
    np.testing.assert_array_equal(players.root_valid[0, trace.frames], trace.valid)
    np.testing.assert_array_equal(players.heading_valid, players.root_valid)
    assert players.root_valid[0, 1] and not players.smpl_valid[0, 1]
    assert players.smpl_valid[0, [0, 2, 4, 5]].all()
    assert players.mesh_reprojection_px[0, 1] == pytest.approx(21.0, abs=1e-3)
    np.testing.assert_allclose(
        players.mesh_reprojection_px[0, [0, 2, 4, 5]], 0, atol=1e-3
    )
    assert players.vertices_local is not None
    for value in (
        players.body_pose[0, 1],
        players.global_orient[0, 1],
        players.vertices_local[0, 1],
    ):
        np.testing.assert_array_equal(value, 0)


def test_trace_reports_missing_and_rejected_frames() -> None:
    case = _build_case()
    trace = _trace(case)
    expected = [
        0,
        0,
        0,
        PlacementRejection.INSUFFICIENT_JOINTS,
        0,
        0,
        PlacementRejection.UNOBSERVABLE_YAW,
        PlacementRejection.UNOBSERVABLE_YAW,
        PlacementRejection.INSUFFICIENT_JOINTS,
        PlacementRejection.UNOBSERVABLE_YAW,
        PlacementRejection.UNOBSERVABLE_YAW,
        PlacementRejection.UNOBSERVABLE_YAW,
    ]
    np.testing.assert_array_equal(trace.reasons, expected)
    np.testing.assert_array_equal(trace.valid, np.array(expected) == 0)
    np.testing.assert_array_equal(trace.hip_position[~trace.valid], 0)
    np.testing.assert_array_equal(trace.yaw_correction[~trace.valid], 0)
    players = _reconstruct(case)
    np.testing.assert_array_equal(players.root_reasons[0, trace.frames], expected)
    np.testing.assert_array_equal(players.root_reasons[0, [6, 7, 14, 15]], 1)


def test_trace_target_joints_use_nan_for_invalid_joints() -> None:
    case = _build_case()
    trace = _trace(case)
    expected_valid = case.skeleton.valid[0, trace.frames]
    assert not expected_valid.all()
    assert np.isnan(trace.target_joints[~expected_valid]).all()
    np.testing.assert_array_equal(
        trace.target_joints[expected_valid],
        case.skeleton.positions[0, trace.frames][expected_valid],
    )
    expected_weights = np.tile([45 / 116] * 5 + [45 / 58] * 12, (len(trace.frames), 1))
    expected_weights[~expected_valid] = 0
    np.testing.assert_allclose(trace.target_weights, expected_weights, atol=1e-7)


def test_trace_does_not_unload_or_mutate_inputs() -> None:
    case = _build_case()
    arrays = [
        case.skeleton.positions,
        case.skeleton.valid,
        case.skeleton.inliers,
        case.skeleton.reprojection_px,
        case.grouped.uv_px,
        case.grouped.confidence,
        case.grouped.visibility,
    ]
    for segment in case.recovered.bodies[0].segments:
        arrays.extend(
            [
                segment.source_frames,
                segment.parameters.body_pose,
                segment.parameters.global_orient,
                segment.parameters.betas,
                segment.parameters.transl,
            ]
        )
    originals = [array.copy() for array in arrays]
    trace = _trace(case)
    assert not case.geometry.unloaded
    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)
    assert trace.valid.any()


def test_speed_rejection_is_visible_and_matches_reconstruction() -> None:
    case = _build_case()
    case.skeleton.positions[0, 2] += np.array([20.0, 0, 0], np.float32)
    trace = _trace(case)
    players = _reconstruct(case)
    np.testing.assert_array_equal(trace.frames[trace.reasons == 6], [0, 1, 2])
    assert not trace.valid[trace.reasons == 6].any()
    np.testing.assert_array_equal(trace.valid, players.root_valid[0, trace.frames])
    np.testing.assert_array_equal(trace.reasons, players.root_reasons[0, trace.frames])
    np.testing.assert_array_equal(players.position[0, [0, 1, 2]], 0)
    np.testing.assert_array_equal(players.smpl_valid[0, [0, 1, 2]], False)
    assert (np.linalg.norm(trace.hip_position[trace.reasons == 6], axis=-1) > 0).all()


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
    assert not case.geometry.calls
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
