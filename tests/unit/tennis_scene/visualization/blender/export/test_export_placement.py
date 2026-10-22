"""Placement bundle exporter: shapes, NaN semantics and rejection paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.submodules.configuration import SmplAssetPaths
from src.tennis_scene.motion_alignment.mesh_placement import (
    canonicalize_incam_body,
    interpolate_rotations,
)
from src.tennis_scene.motion_alignment.temporal import TemporalPlacementConfig
from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyParameters
from src.tennis_scene.pipeline.components.gvhmr import (
    GVHMROutput,
    RecoveredBody,
    RecoveredBodySegment,
)
from src.tennis_scene.pipeline.components.player_reconstruction import (
    PlayerSkeleton,
    reconstruct_player_bodies,
    trace_player_placement,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
)
from src.tennis_scene.visualization.blender.export.placement import (
    PLACEMENT_BUNDLE_SCHEMA,
    PLACEMENT_SENSOR_WIDTH_MM,
    PLACEMENT_TOLERANCE_M,
    PlacementExportConfig,
    PlacementInputs,
    _assert_world_matches,
    _asset_binding,
    _read_artifact,
    export_placement_bundle,
)
from src.tennis_scene.visualization.blender.export.pointcache import read_pc2
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.writer import save_video_rgb

FRAMES = 16
VERTICES = 12
IDENTITY = 7
FPS = 30.0
SIZE = (1280, 720)


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

    @property
    def root_regressor(self) -> NDArray[np.float64]:
        return self._root

    def reconstruct(self, parameters: BodyParameters) -> BodyGeometry:
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


def _cameras() -> tuple[PinholeCamera, ...]:
    intrinsic = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
    )
    first = PinholeCamera("cam0", intrinsic, np.eye(3), np.array([0.0, 0.0, 8.0]))
    second = PinholeCamera(
        "cam1",
        intrinsic,
        Rotation.from_euler("y", 0.4).as_matrix(),
        np.array([0.5, 0.0, 8.0]),
    )
    return first, second


def _segments(adjacent: bool = False) -> tuple[RecoveredBodySegment, ...]:
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
    sources: tuple[NDArray[np.int64], ...]
    if adjacent:
        sources = (
            np.arange(0, 8, dtype=np.int64),
            np.arange(8, 16, dtype=np.int64),
        )
    else:
        sources = (
            np.arange(0, 6, dtype=np.int64),
            np.arange(8, 14, dtype=np.int64),
        )
    return tuple(
        RecoveredBodySegment(
            source,
            BodyParameters(
                body_pose=body_pose[source],
                global_orient=orient[source],
                betas=betas[source],
                transl=transl[source],
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


@dataclass(frozen=True)
class _Case:
    inputs: PlacementInputs
    config: PlacementExportConfig
    geometry: _FakeGeometry
    faces_path: Path
    video_path: Path
    skeleton: PlayerSkeleton
    grouped: GroupedObservations
    trace_valid: NDArray[np.bool_]
    trace_frames: NDArray[np.int64]


def _dense_parameters(
    params: BodyParameters, source: NDArray[np.int64], selected: NDArray[np.int64]
) -> BodyParameters:
    return BodyParameters(
        interpolate_rotations(params.body_pose, source, selected),
        interpolate_rotations(params.global_orient, source, selected),
        np.repeat(params.betas[:1], len(selected), axis=0),
        np.stack(
            [np.interp(selected, source, params.transl[:, column]) for column in range(3)],
            -1,
        ).astype(np.float32),
    )


def _build_case(
    tmp_path: Path, *, adjacent: bool = False, t0: int = 2
) -> _Case:
    geometry = _FakeGeometry()
    cameras = _cameras()
    segments = _segments(adjacent)
    valid = _valid_mask()
    canonical_frames: NDArray[np.float64] = np.zeros((FRAMES, 17, 3), np.float64)
    for segment in segments:
        source = segment.source_frames
        dense: NDArray[np.int64] = np.arange(
            int(source[0]), int(source[-1]) + 1, dtype=np.int64
        )
        params = _dense_parameters(segment.parameters, source, dense)
        body = geometry.reconstruct(params)
        canonical = canonicalize_incam_body(
            body.vertices, body.coco17, params.global_orient, cameras[0], geometry.root_regressor
        )
        canonical_frames[source] = canonical.joints
    true_scale = 1.12
    target: NDArray[np.float32] = np.zeros((FRAMES, 17, 3), np.float32)
    for frame in range(8):
        angle = 0.35 + 0.06 * frame
        root = np.array([1.2 + 0.05 * frame, -1.5 + 0.08 * frame, 0.8])
        correction = Rotation.from_euler("z", angle).as_matrix()
        target[frame] = true_scale * (correction @ canonical_frames[frame].T).T + root
    target[8:14] = np.array([2.0, -0.5, 1.0], np.float32)
    target[~valid] = 0

    uv: NDArray[np.float32] = np.zeros((1, 2, FRAMES, 17, 2), np.float32)
    for camera_index, camera in enumerate(cameras):
        projected, front = camera.project(target.astype(np.float64))
        assert front[valid].all()
        uv[0, camera_index, valid] = projected[valid].astype(np.float32)
    confidence = np.broadcast_to(
        np.where(valid, 0.9, 0.0)[None, None], (1, 2, FRAMES, 17)
    ).copy()
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
        SIZE,
        FPS,
        np.zeros((2, FRAMES, 1, 17, 2), np.float32),
        np.zeros((2, FRAMES, 1, 17), np.float32),
        np.ones((2, FRAMES, 1), bool),
        np.zeros((2, 1), np.int64),
        np.zeros((2, FRAMES, 1, 3), np.float32),
    )
    recovered = GVHMROutput((RecoveredBody(IDENTITY, "cam0", segments),))
    config = TemporalPlacementConfig()
    placed = reconstruct_player_bodies(
        grouped,
        raw,
        skeleton,
        cameras,
        recovered,
        body=geometry,
        reprojection_px=1e9,
        placement_config=config,
    )
    geometry.unloaded = False
    trace = trace_player_placement(
        recovered.bodies[0],
        skeleton,
        grouped,
        cameras,
        player=0,
        body=geometry,
        fps=FPS,
        placement_config=config,
    )

    video_path = tmp_path / "cam0.mp4"
    save_video_rgb(np.zeros((FRAMES, 720, 1280, 3), np.uint8), video_path, fps=FPS)
    source = ClipSource(
        "synthetic",
        (
            SourceVideo("cam0", video_path, dual_sha256(video_path), FRAMES, FPS, *SIZE),
            SourceVideo(
                "cam1", tmp_path / "cam1.mp4", "0" * 64, FRAMES, FPS, *SIZE
            ),
        ),
    )
    joint_regressor: NDArray[np.float32] = np.zeros((1, VERTICES), np.float32)
    joint_regressor[0, 0] = 1.0
    assets = SmplAssetPaths(
        hmr2_mean_params=tmp_path / "hmr2_mean_params.npz",
        smplx_to_smpl=tmp_path / "smplx_to_smpl.pt",
        smpl_coco17_regressor=tmp_path / "smpl_coco17.pt",
        smplx_verts437=tmp_path / "smplx_verts437.pt",
        smpl_neutral_joint_regressor=tmp_path / "joint_regressor.pt",
    )
    faces_path = tmp_path / "faces.npz"
    np.savez(faces_path, f=np.array([[0, 1, 2], [1, 2, 3]], np.int64))
    inputs = PlacementInputs(
        source=source,
        cameras=cameras,
        grouped=grouped,
        skeleton=skeleton,
        recovered=recovered,
        placed=placed,
        placement_config=config,
        joint_regressor=joint_regressor,
        body_assets=assets,
        body_models_dir=tmp_path / "body_models",
        asset_provenance={},
        artifact_provenance={},
        store_provenance={},
    )
    export_config = PlacementExportConfig(
        identity=IDENTITY,
        frames=(0, 6),
        t0=t0,
        plate_camera_id="cam0",
        plate_video_path=video_path,
        human_vis_threshold=0.15,
    )
    return _Case(
        inputs,
        export_config,
        geometry,
        faces_path,
        video_path,
        skeleton,
        grouped,
        np.asarray(trace.valid),
        np.asarray(trace.frames),
    )


def _export(case: _Case, output_dir: Path) -> Path:
    return export_placement_bundle(
        case.inputs,
        body=case.geometry,
        output_dir=output_dir,
        config=case.config,
        smpl_faces_path=case.faces_path,
        resolved_config={"mode": "export", "identity": IDENTITY},
        git_commit="a" * 40,
    )


def test_export_writes_schema_shapes_and_nan(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    output = tmp_path / "bundle"
    manifest_path = _export(case, output)

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == PLACEMENT_BUNDLE_SCHEMA
    assert manifest["fps"] == FPS
    assert manifest["frames"]["start"] == 0
    assert manifest["frames"]["stop"] == 6
    assert manifest["t0"] == 2
    assert manifest["identity"] == IDENTITY
    assert manifest["gvhmr_camera_id"] == "cam0"
    assert manifest["scale"] > 0
    assert manifest["coco17_skeleton"] == [list(edge) for edge in COCO17_SKELETON]
    assert set(manifest["court"]) == {"ground_lines", "runoff_xy", "net"}
    assert manifest["provenance"]["data_origin"] == "measured"
    assert manifest["provenance"]["git_commit"] == "a" * 40
    assert manifest["provenance"]["resolved_config"]["mode"] == "export"
    assert manifest["plate"]["camera_id"] == "cam0"
    assert manifest["plate"]["video_sha256"] == dual_sha256(case.video_path)

    body = read_pc2(output / "body_local.pc2")[0]
    assert body.shape == (6, VERTICES, 3)
    assert body.dtype == np.float32
    valid = np.load(output / "valid.npy")
    assert valid.shape == (6,)
    assert valid.dtype == np.bool_
    assert not valid.all()
    assert np.isfinite(body).all()
    np.testing.assert_array_equal(body[~valid], 0.0)

    joints = np.load(output / "joints_local.npy")
    assert joints.shape == (6, 17, 3)
    assert joints.dtype == np.float64
    target = np.load(output / "target_joints.npy")
    assert target.shape == (6, 17, 3)
    assert target.dtype == np.float32
    weights = np.load(output / "target_weights.npy")
    assert weights.shape == (6, 17)
    assert weights.dtype == np.float64
    hip = np.load(output / "hip_position.npy")
    assert hip.shape == (6, 3)
    assert hip.dtype == np.float64
    yaw = np.load(output / "yaw_correction.npy")
    assert yaw.shape == (6,)
    assert yaw.dtype == np.float64
    t0_vertices = np.load(output / "placed_vertices_t0.npy")
    assert t0_vertices.shape == (VERTICES, 3)
    assert t0_vertices.dtype == np.float32
    faces = np.load(output / "smpl_faces.npy")
    assert faces.shape == (2, 3)
    assert faces.dtype == np.int32
    assert sorted(path.name for path in (output / "plate").iterdir()) == [
        f"{frame:05d}.jpg" for frame in range(6)
    ]

    # target_joints keeps NaN for invalid triangulation joints and never a
    # fabricated coordinate; target_weights is zero there.
    expected_valid = case.skeleton.valid[0, case.trace_frames[:6]]
    assert np.isnan(target[~expected_valid]).all()
    assert (weights[~expected_valid] == 0).all()
    assert np.isfinite(target[expected_valid]).all()


def test_export_manifest_records_plate_camera_calibration(tmp_path: Path) -> None:
    case = _build_case(tmp_path)

    import json

    manifest = json.loads(_export(case, tmp_path / "bundle").read_text())
    camera = case.inputs.cameras[0]
    assert manifest["width"] == SIZE[0]
    assert manifest["height"] == SIZE[1]
    assert manifest["plate"]["camera_id"] == "cam0"
    np.testing.assert_array_equal(
        manifest["camera"]["opencv"]["K"], camera.intrinsic
    )
    np.testing.assert_array_equal(
        manifest["camera"]["opencv"]["R"], camera.rotation
    )
    np.testing.assert_array_equal(
        manifest["camera"]["opencv"]["t"], camera.translation
    )
    blender = manifest["camera"]["blender"]
    assert blender["resolution_x"] == SIZE[0]
    assert blender["resolution_y"] == SIZE[1]
    assert blender["sensor_fit"] == "HORIZONTAL"
    assert blender["sensor_width_mm"] == PLACEMENT_SENSOR_WIDTH_MM
    assert blender["lens_mm"] == pytest.approx(
        camera.intrinsic[0, 0] * PLACEMENT_SENSOR_WIDTH_MM / SIZE[0]
    )
    np.testing.assert_allclose(
        blender["location"], -camera.rotation.T @ camera.translation
    )


def test_export_uses_the_selected_plate_camera(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    cam1_video = tmp_path / "cam1.mp4"
    frames: NDArray[np.uint8] = np.ones((FRAMES, 720, 1280, 3), np.uint8)
    save_video_rgb(frames, cam1_video, fps=FPS)
    source = case.inputs.source
    videos = (
        source.videos[0],
        SourceVideo(
            "cam1",
            cam1_video,
            dual_sha256(cam1_video),
            FRAMES,
            FPS,
            *SIZE,
        ),
    )
    inputs = replace(
        case.inputs,
        source=ClipSource(source.clip_id, videos),
    )
    config = replace(case.config, plate_camera_id="cam1", plate_video_path=cam1_video)

    import json

    manifest = json.loads(
        export_placement_bundle(
            inputs,
            body=case.geometry,
            output_dir=tmp_path / "bundle",
            config=config,
            smpl_faces_path=case.faces_path,
        ).read_text()
    )
    selected = inputs.cameras[1]
    np.testing.assert_array_equal(
        manifest["camera"]["opencv"]["K"], selected.intrinsic
    )
    np.testing.assert_array_equal(
        manifest["camera"]["opencv"]["R"], selected.rotation
    )
    assert manifest["plate"]["camera_id"] == "cam1"


def test_export_rejects_uncalibrated_plate_camera(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    # cam2 is a source camera but the fixture has no calibration for it.
    source = case.inputs.source
    cam2_video = tmp_path / "cam2.mp4"
    videos = source.videos + (
        SourceVideo("cam2", cam2_video, "0" * 64, FRAMES, FPS, *SIZE),
    )
    inputs = replace(case.inputs, source=ClipSource(source.clip_id, videos))
    config = replace(case.config, plate_camera_id="cam2")
    with pytest.raises(ValueError, match="has no calibrated camera"):
        export_placement_bundle(
            inputs,
            body=case.geometry,
            output_dir=tmp_path / "bundle",
            config=config,
            smpl_faces_path=case.faces_path,
        )
    assert not (tmp_path / "bundle").exists()


def test_export_accepts_a_window_that_starts_later(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    config = replace(case.config, frames=(2, 6), t0=4)
    manifest_path = export_placement_bundle(
        case.inputs,
        body=case.geometry,
        output_dir=tmp_path / "bundle",
        config=config,
        smpl_faces_path=case.faces_path,
    )

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["frames"]["source_frames"] == [2, 3, 4, 5]
    assert manifest["t0"] == 4
    assert np.load(tmp_path / "bundle" / "valid.npy").shape == (4,)
    assert np.load(tmp_path / "bundle" / "placed_vertices_t0.npy").shape == (
        VERTICES,
        3,
    )


def test_export_rejects_existing_output(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _export(case, output)


def test_export_rejects_invalid_t0(tmp_path: Path) -> None:
    # Frame 3 is in the first GVHMR segment but rejected for insufficient joints.
    case = _build_case(tmp_path, t0=3)
    with pytest.raises(ValueError, match="t0=3 is not valid"):
        _export(case, tmp_path / "bundle")


def test_export_rejects_frames_that_cross_segments(tmp_path: Path) -> None:
    case = _build_case(tmp_path, adjacent=True)
    config = replace(case.config, frames=(4, 12), t0=4)
    with pytest.raises(ValueError, match="cross GVHMR segments"):
        export_placement_bundle(
            case.inputs,
            body=case.geometry,
            output_dir=tmp_path / "bundle",
            config=config,
            smpl_faces_path=case.faces_path,
        )


def test_export_rejects_frames_outside_recovered_segment(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    config = replace(case.config, frames=(4, 10), t0=4)
    with pytest.raises(ValueError, match="missing source frames"):
        export_placement_bundle(
            case.inputs,
            body=case.geometry,
            output_dir=tmp_path / "bundle",
            config=config,
            smpl_faces_path=case.faces_path,
        )


def test_export_rejects_a_saved_placement_that_disagrees(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    placed = case.inputs.placed
    assert placed.vertices_local is not None
    placed.vertices_local[0, 2, 0] += 1.0
    with pytest.raises(ValueError, match="differs from the saved body_placement"):
        _export(case, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_export_rejects_non_finite_saved_placement(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    placed = case.inputs.placed
    assert placed.vertices_local is not None
    placed.vertices_local[0, 2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="saved world position is non-finite") as error:
        _export(case, tmp_path / "bundle")
    message = str(error.value)
    assert "identity=7" in message
    assert "source_frame=2" in message
    assert "vertex=" in message
    assert not (tmp_path / "bundle").exists()


def test_assert_world_matches_rejects_non_finite_recomputed() -> None:
    recomputed: NDArray[np.float64] = np.zeros((1, 2, 3), np.float64)
    saved: NDArray[np.float64] = np.zeros((1, 2, 3), np.float64)
    recomputed[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="recomputed world position is non-finite"):
        _assert_world_matches(
            recomputed,
            saved,
            np.ones(1, bool),
            np.asarray([5], np.int64),
            IDENTITY,
            context="Recomputed placement",
        )


def test_assert_world_matches_tolerance_is_inclusive() -> None:
    recomputed: NDArray[np.float64] = np.zeros((1, 2, 3), np.float64)
    saved: NDArray[np.float64] = np.zeros((1, 2, 3), np.float64)
    mask: NDArray[np.bool_] = np.ones(1, bool)
    frames = np.asarray([3], np.int64)
    saved[0, 0, 0] = PLACEMENT_TOLERANCE_M
    with pytest.raises(ValueError, match="by at least"):
        _assert_world_matches(
            recomputed, saved, mask, frames, IDENTITY, context="Recomputed placement"
        )
    saved[0, 0, 0] = np.nextafter(PLACEMENT_TOLERANCE_M, 0.0)
    _assert_world_matches(
        recomputed, saved, mask, frames, IDENTITY, context="Recomputed placement"
    )


def test_export_rejects_root_validity_mismatch(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    case.inputs.placed.root_valid[0, 1] = False
    with pytest.raises(ValueError, match="root_valid disagrees") as error:
        _export(case, tmp_path / "bundle")
    message = str(error.value)
    assert "identity=7" in message
    assert "source_frames=[1]" in message
    assert "saved_root_valid=[False]" in message
    assert "trace_valid=[True]" in message
    assert not (tmp_path / "bundle").exists()


def test_export_rejects_missing_saved_body_at_t0(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    case.inputs.placed.smpl_valid[0, 2] = False
    with pytest.raises(ValueError, match="has no saved SMPL body"):
        _export(case, tmp_path / "bundle")


def test_export_rejects_plate_video_hash_mismatch(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    other = tmp_path / "other.mp4"
    frames: NDArray[np.uint8] = np.zeros((FRAMES, 720, 1280, 3), np.uint8)
    frames[..., 0] = 7
    save_video_rgb(frames, other, fps=FPS)
    config = replace(case.config, plate_video_path=other)
    with pytest.raises(ValueError, match="Plate video SHA-256 mismatch"):
        export_placement_bundle(
            case.inputs,
            body=case.geometry,
            output_dir=tmp_path / "bundle",
            config=config,
            smpl_faces_path=case.faces_path,
        )


def test_target_joints_nan_comes_from_trace_not_zeros(tmp_path: Path) -> None:
    case = _build_case(tmp_path)
    trace = trace_player_placement(
        case.inputs.recovered.bodies[0],
        case.skeleton,
        case.grouped,
        case.inputs.cameras,
        player=0,
        body=case.geometry,
        fps=FPS,
        placement_config=case.inputs.placement_config,
    )
    invalid = ~case.skeleton.valid[0, trace.frames]
    assert np.isnan(trace.target_joints[invalid]).all()


class _StubReader:
    def __init__(self, references: Mapping[str, Mapping[str, Any]]) -> None:
        self.root = Path("/nonexistent")
        self.index_path = Path("/nonexistent/scene.json")
        self.source: Mapping[str, Any] = {"clip_id": "x", "videos": []}
        self.source_sha256 = "0" * 64
        self.camera_ids: tuple[str, ...] = ()
        self.references: Mapping[str, Mapping[str, Any]] = references
        self.stale: tuple[str, ...] = ()

    def payload(self, node: str) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {"arrays": {}}

    def stale_dependencies(self, node: str) -> tuple[str, ...]:
        return self.stale


def test_read_artifact_rejects_old_person_identities_version() -> None:
    reader = _StubReader(
        {
            "player_association": {
                "artifact_id": "a",
                "path": "x.json",
                "sha256": "b",
                "schema": "person_identities",
                "version": 2,
                "execution_key": "c",
            }
        }
    )
    with pytest.raises(ValueError, match="requires person_identities:3"):
        _read_artifact(
            reader,
            "player_association",
            schema="person_identities",
            version=3,
            output_type=dict,
        )


def test_read_artifact_rejects_superseded_dependencies() -> None:
    reader = _StubReader(
        {
            "body_placement": {
                "artifact_id": "a",
                "path": "x.json",
                "sha256": "b",
                "schema": "placed_bodies",
                "version": 1,
                "execution_key": "c",
            }
        }
    )
    reader.stale = ("alignment: upstream artifact was superseded",)
    with pytest.raises(ValueError, match="superseded dependencies"):
        _read_artifact(
            reader,
            "body_placement",
            schema="placed_bodies",
            version=1,
            output_type=dict,
        )


def test_asset_binding_rejects_sha_mismatch(tmp_path: Path) -> None:
    assets: dict[str, dict[str, str]] = {}
    for key, name in {
        "body_model": "SMPLX_NEUTRAL.npz",
        "smpl_hmr2_mean_params": "smpl_mean_params.npz",
        "smpl_smplx_to_smpl": "smplx2smpl_sparse.pt",
        "smpl_smpl_coco17_regressor": "smpl_coco17_J_regressor.pt",
        "smpl_smplx_verts437": "smplx_verts437.pt",
        "smpl_smpl_neutral_joint_regressor": "smpl_neutral_J_regressor.pt",
    }.items():
        parent = tmp_path / "body_models" / "smplx" if key == "body_model" else tmp_path
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / name
        path.write_bytes(name.encode())
        digest = dual_sha256(path)
        if key == "smpl_smplx_to_smpl":
            digest = "0" * 64
        assets[key] = {"path": str(path), "sha256": digest}
    descriptor = {"identity": {"settings": {"assets": assets, "config": {}}}}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _asset_binding(descriptor)
