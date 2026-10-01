"""Headless end-to-end orchestration with known cameras and deterministic predictors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionModule,
    BallDetectionOutput,
)
from src.tennis_scene.pipeline.components.court_kp import CourtKPModule, CourtKPResult
from src.tennis_scene.pipeline.contracts import ComponentIO
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.configuration import PathRole
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import CourtConfig, court_keypoints_3d
from src.utils.video import VideoInfo


def _camera(name: str, center: list[float]) -> PinholeCamera:
    c = np.asarray(center, np.float64)
    forward = np.array([0., 0., 1.]) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    r = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]]), r, -r @ c)


def runtime(tmp_path: Path, *, execute_identity_overrides: bool = False, overrides: tuple[str, ...] = ()) -> PipelineRuntimeConfig:
    """``execute_identity_overrides`` executes test stand-ins for the import-only nodes."""
    config_dir = Path(__file__).parents[4] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="pipeline", overrides=[
            "device=cpu", "gvhmr.enabled=false", f"paths.data_root={tmp_path}",
            f"paths.artifact_root={tmp_path}", f"paths.output_root={tmp_path}",
            f"paths.checkpoint_root={tmp_path / 'ckpt'}", f"paths.external_asset_root={tmp_path / 'third_party'}",
            "output_directory=run", "cache.directory=stages",
            *(["execution.player_association=execute", "execution.court_side=execute"] if execute_identity_overrides else []),
            *overrides,
        ])
    return PipelineRuntimeConfig.from_config(cfg)


def materialize_assets(paths: Any, tmp_path: Path) -> None:
    """Create stand-ins for the test-owned model assets under ``tmp_path``.

    Repository-bundled assets (``src/submodules/vendor``) resolve outside
    ``tmp_path``; they are real files and must never be overwritten by a test.
    """
    for path in paths:
        if path.is_relative_to(tmp_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"asset")
        else:
            assert path.is_file(), f"repository-bundled asset is missing: {path}"


class FixedStage:
    io: ComponentIO[Any, Any]
    def __init__(self, result: Any) -> None:
        self.result, self.calls = result, 0
    def process(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.result
    def unload(self) -> None:
        pass


def inputs(*, empty: bool = False, frames: int = 24) -> tuple[CourtKPResult, ObjectObservations, tuple[BallDetectionOutput, ...]]:
    cameras = (_camera("cam0", [-8., -16., 9.]), _camera("cam1", [8., -16., 9.]), _camera("cam2", [5., 16., 9.]))
    local = (cameras[0], cameras[1], cameras[2].half_turned(True))
    template = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14]
    court_px = np.stack([c.project(template)[0] for c in local])
    kp = np.repeat((court_px / [1279, 719])[:, None], frames, axis=1).astype(np.float32)
    vis = np.repeat(((court_px >= 0) & (court_px < [1280, 720])).all(-1)[:, None], frames, axis=1).astype(np.float32)
    diagnostics: dict[str, Any] = {"output_keypoint_contract": "camera_view_v2", "cameras": []}
    for camera in local:
        h = camera.intrinsic @ np.column_stack((camera.rotation[:, 0], camera.rotation[:, 1], camera.translation))
        diagnostics["cameras"].append({"frames": [{"frame_index": t, "status": "ok", "homography_court_metres_to_image_pixels": h.tolist()} for t in range(frames)]})
    court = CourtKPResult(kp, vis, np.arange(frames, dtype=np.int32), diagnostics)
    people_xyz: NDArray[np.float64] = np.zeros((frames, 17, 3), np.float64)
    people_xyz[..., 0] = np.linspace(-.2, .2, 17)
    people_xyz[..., 1] = -4 + np.arange(frames)[:, None] * .01
    people_xyz[..., 2] = np.linspace(.5, 1.8, 17)
    human = np.stack([c.project(people_xyz)[0] for c in cameras]).astype(np.float32)[:, :, None]
    confidence = np.full(human.shape[:-1], .9, np.float32)
    observed = np.ones(human.shape[:3], bool)
    ball_xyz = np.c_[np.zeros(frames), -3 + np.arange(frames) * .01, np.ones(frames) * 1.4]
    ball_px = np.stack([c.project(ball_xyz)[0] for c in cameras]).astype(np.float32)
    ball_visible: NDArray[np.bool_] = np.ones((3, frames), bool)
    score: NDArray[np.float32] = np.full((3, frames), .9, np.float32)
    if empty:
        human, confidence, observed = human[:, :, :0], confidence[:, :, :0], observed[:, :, :0]
        ball_px[:] = 0
        ball_visible[:] = False
        score[:] = 0
    people = ObjectObservations(tuple(c.camera_id for c in cameras), (1280, 720), 30., human, confidence, observed, np.zeros((3, human.shape[2]), np.int64))
    balls = tuple(BallDetectionOutput(camera.camera_id, np.arange(frames, dtype=np.int64), ball_px[v], score[v], ball_visible[v],
                                      ball_visible[v].astype(np.uint8), "model_score") for v, camera in enumerate(cameras))
    return court, people, balls


def camera_stages(court: CourtKPResult, people: ObjectObservations, balls: tuple[BallDetectionOutput, ...]) -> dict[str, FixedStage]:
    """Fixed outputs of every camera-scoped node (court, ball, person detection/tracking/pose)."""
    from src.tennis_scene.pipeline.components.person_detection import (
        PersonDetectionModule,
        PersonDetectionOutput,
    )
    from src.tennis_scene.pipeline.components.person_tracking import (
        PersonTrackingModule,
        PersonTrackingOutput,
    )
    from src.tennis_scene.pipeline.components.pose_estimation import (
        PoseEstimationModule,
    )
    assert court.diagnostics is not None
    frames = people.num_frames
    stages: dict[str, FixedStage] = {}
    def fixed(name: str, io: ComponentIO[Any, Any], result: Any) -> None:
        stage = FixedStage(result)
        stage.io = io
        stages[name] = stage
    for v, camera in enumerate(people.camera_ids):
        local_court = CourtKPResult(court.keypoints[v:v+1, :1], court.visibility[v:v+1, :1], np.array([0], np.int32),
            {"output_keypoint_contract": "camera_view_v2", "temporal_policy": "static_first_frame",
             "cameras": [{"frames": [court.diagnostics["cameras"][v]["frames"][0]]}]})
        fixed(f"court_detection/{camera}", CourtKPModule.io, local_court)
        fixed(f"ball_detection/{camera}", BallDetectionModule.io, balls[v])
        count = people.uv_px.shape[2]
        boxes = np.zeros((count, frames, 4), np.float32)
        boxes[..., 2:] = 100
        fixed(f"person_detection/{camera}", PersonDetectionModule.io, PersonDetectionOutput(camera,
            np.arange(frames + 1, dtype=np.int64) * count, boxes.transpose(1, 0, 2).reshape(-1, 4), np.ones(frames * count, np.float32)))
        fixed(f"person_tracking/{camera}", PersonTrackingModule.io, PersonTrackingOutput(camera,
            np.arange(count, dtype=np.int64), boxes, people.observed[v].T,
            tuple((i,) for i in range(count)), ()))
        poses = people.select_views((v,))
        pose_boxes = np.zeros((*poses.observed.shape, 3), np.float32)
        pose_boxes[..., 2] = 100
        fixed(f"pose_estimation/{camera}", PoseEstimationModule.io, replace(poses, boxes_xys=pose_boxes))
    return stages


def patch_video_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frames: int = 24) -> tuple[Path, ...]:
    """Three fixture video files whose probe reports 1280x720 at 30 fps."""
    paths = tuple(tmp_path / f"cam{i}.mp4" for i in range(3))
    for path in paths:
        path.write_bytes(b"video fixture")
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info", lambda _: VideoInfo(30., 1280, 720, frames))
    return paths


def setup_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, empty: bool = False) -> tuple[TennisSceneOrchestrator, tuple[Path, ...], dict[str, Any]]:
    from src.tennis_scene.pipeline.components.identity import (
        CourtSideOutput,
        DeclaredArtifacts,
        PlayerIdentitiesOutput,
        court_side_io,
        player_association_io,
    )
    from src.tennis_scene.pipeline.input_assembly.observations import gather_people
    cfg = runtime(tmp_path, execute_identity_overrides=True)
    court, people, balls = inputs(empty=empty)
    stages: dict[str, Any] = dict(camera_stages(court, people, balls))
    class Identities:
        """Stands in for the imported association: equal tracker IDs are one player."""
        io = player_association_io(people.camera_ids)
        calls = 0
        def process(self, request: DeclaredArtifacts) -> PlayerIdentitiesOutput:
            self.calls += 1
            calibration = request.artifacts["calibration"]
            active = tuple(v.source_index for v in calibration.calibration.views)
            raw = gather_people(request.context.source, request.artifacts).select_views(active)
            return PlayerIdentitiesOutput(raw.camera_ids, raw.local_track_ids.copy(), raw.local_track_ids.copy())
    class Side:
        io = court_side_io(people.camera_ids)
        calls = 0
        def process(self, request: DeclaredArtifacts) -> CourtSideOutput:
            self.calls += 1
            calibration = request.artifacts["calibration"]
            return CourtSideOutput(calibration.calibration.camera_ids, calibration.reference_camera, (False, False, True))
    stages["player_association"], stages["court_side"] = Identities(), Side()
    pipeline = TennisSceneOrchestrator(cfg, components=stages)
    paths = patch_video_probe(tmp_path, monkeypatch)
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Automatic pipeline attempted human interaction")
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(cv2, "namedWindow", forbidden)
    return pipeline, paths, stages


def test_headless_declared_pipeline_and_disk_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    assert scene.schema_version == 2 and scene.num_frames == 24
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]
    assert scene.player_kp_3d_vis.all() and scene.ball_3d_valid.all()
    assert all(stage.calls == 1 for stage in stages.values())
    assert pipeline.last_store is not None
    assert pipeline.last_store.index_path.is_file()
    assert not scene.player_valid.any()
    pipeline.config = replace(pipeline.config, cache_source="load")
    again = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    np.testing.assert_array_equal(again.ball_3d, scene.ball_3d)
    assert all(stage.calls == 1 for stage in stages.values())
    assert pipeline.last_runner is not None
    assert set(pipeline.last_runner.statuses.values()) == {"loaded"}
    output = tmp_path / "roundtrip.npz"
    save_scene_result(again, output)
    np.testing.assert_array_equal(load_scene_result(output).ball_3d, scene.ball_3d)


def test_empty_observations_preserve_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch, empty=True)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    assert scene.metadata["status"] == "empty"
    assert scene.metadata["court_reference"] is None
    assert scene.player_position.shape == (0, 24, 3)
    assert not scene.ball_3d_valid.any()


def test_single_ball_missing_views_remain_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    for camera in ("cam1", "cam2"):
        result = stages[f"ball_detection/{camera}"].result
        result.observed[7] = False
        result.point_kind[7] = 0
        result.confidence[7] = 0
        result.uv_px[7] = 0
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    assert not scene.ball_3d_valid[7] and scene.ball_3d_valid.sum() == 23
    assert not scene.ball_3d[7].any()


def test_v2_missing_mask_rejected_before_archive_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch, empty=True)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    scene.ball_3d_valid = None
    with pytest.raises(ValueError, match="ball_3d_valid"):
        save_scene_result(scene, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("missing_hips", [False, True])
def test_selected_gvhmr_parameters_are_placed_and_resumed_without_reinference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_hips: bool) -> None:
    from scipy.spatial.transform import Rotation

    from src.tennis_scene.pipeline.body_types import BodyGeometry, BodyParameters
    from src.tennis_scene.pipeline.components.gvhmr import (
        GVHMRModule,
        GVHMROutput,
        RecoveredBody,
        RecoveredBodySegment,
    )
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    if missing_hips:
        for camera in ("cam0", "cam1", "cam2"):
            stage = stages[f"pose_estimation/{camera}"]
            confidence = stage.result.confidence.copy()
            confidence[..., 11:13] = 0
            stage.result = replace(stage.result, confidence=confidence)
    cam = _camera("cam0", [-8., -16., 9.])
    class KnownBody:
        root_regressor = np.array([1., 0, 0, 0])
        calls = 0
        unloaded = False
        def recover(self, req):
            self.calls += 1
            assert req.source_frames.tolist() == list(range(24))
            assert req.keypoints.shape == (24, 17, 3)
            assert req.intrinsic.shape == (3, 3)
            return BodyParameters(np.zeros((24, 63), np.float32),
                Rotation.from_matrix(cam.rotation @ Rotation.from_euler("z", (.4 + .02 * req.source_frames)[:, None]).as_matrix()).as_rotvec().astype(np.float32),
                np.zeros((24, 10), np.float32),
                np.c_[req.source_frames, np.zeros((24, 2))].astype(np.float32))
        def reconstruct(self, params):
            count = len(params.transl)
            world: NDArray[np.float32] = np.zeros((count, 17, 3), np.float32)
            world[..., 0] = np.linspace(-.2, .2, 17)
            world[..., 1] = -4 + params.transl[:, 0, None] * .01
            world[..., 2] = np.linspace(.5, 1.8, 17)
            hip = world[:, 11:13].mean(1)
            rotation = Rotation.from_euler("z", (.4 + .02 * params.transl[:, 0])[:, None]).as_matrix()
            drift = np.c_[1.5 + .1 * params.transl[:, 0], np.full(count, .2), np.full(count, .1)]
            world = np.einsum("tij,tkj->tki", rotation, world-hip[:, None]) + hip[:, None] + drift[:, None]
            root = hip + drift + rotation @ np.array([.03, 0., -.05])
            verts = root[:, None] + np.einsum("tij,vj->tvi", rotation, np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]]))
            return BodyGeometry((verts @ cam.rotation.T + cam.translation).astype(np.float32), (world @ cam.rotation.T + cam.translation).astype(np.float32))
        def unload(self):
            self.unloaded = True
    body = KnownBody()
    class KnownGVHMR:
        io = GVHMRModule.io
        def process(self, inputs):
            return GVHMROutput(tuple(RecoveredBody(selected.person_id, selected.camera_id,
                tuple(RecoveredBodySegment(request.source_frames, body.recover(request), count)
                    for request, count in zip(selected.requests, selected.observed_samples, strict=True)))
                for selected in inputs.selection.selections))
    stages["gvhmr"] = KnownGVHMR()
    monkeypatch.setattr("src.tennis_scene.pipeline.components.body_placement.SmplGeometry", lambda config: body)
    config = replace(pipeline.config, enabled={**pipeline.config.enabled, "gvhmr": True}, processing_settings={**pipeline.config.processing_settings, "gvhmr": {"enabled": True}})
    materialize_assets(config.people.body_assets().values(), tmp_path)  # identity inputs of the real body_placement
    pipeline = TennisSceneOrchestrator(config, components=stages)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    assert scene.player_valid.all() and scene.player_heading_valid.all() and scene.player_smpl_valid.all()
    assert body.calls == 1 and body.unloaded
    expected = np.column_stack((np.full(24, np.linspace(-.2, .2, 17)[11:13].mean()), -4 + np.arange(24)*.01,
                                np.full(24, np.linspace(.5, 1.8, 17)[11:13].mean()))) + [.03, 0., -.05]
    np.testing.assert_allclose(scene.player_position[0], expected, atol=1e-4)
    np.testing.assert_allclose(scene.player_yaw[0], 0, atol=1e-4)
    np.testing.assert_array_equal(scene.smpl_body_pose, 0)
    assert scene.metadata["body_placement"]["players"]["0"]["scale"] == pytest.approx(1, abs=1e-5)
    if missing_hips:
        assert not scene.player_kp_3d_vis[..., 11:13].any()
    pipeline.config = replace(config, cache_source="load")
    again = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
    np.testing.assert_array_equal(again.player_position, scene.player_position)
    assert body.calls == 1
    pipeline.config = replace(pipeline.config, player_placement=replace(config.player_placement, temporal_weight=.2))
    with pytest.raises(ValueError, match="identity"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=None)
