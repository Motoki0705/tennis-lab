"""Headless end-to-end orchestration with known cameras and deterministic predictors."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray

from src.tasks.plcs.inference.person_predictor import PlayerReIDPredictor
from src.tasks.plcs.model_io.person_association import (
    CourtSideResult,
    PersonInferencePolicy,
    PersonObservationRequest,
    PersonReIDResult,
)
from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.model_io.observations import ObjectObservations
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


def runtime(tmp_path: Path) -> PipelineRuntimeConfig:
    config_dir = Path(__file__).parents[4] / "src/tennis_scene/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="pipeline", overrides=[
            "device=cpu", "gvhmr.enabled=false", f"paths.data_root={tmp_path}",
            f"paths.artifact_root={tmp_path}", f"paths.output_root={tmp_path}",
            "output_directory=run", "cache.directory=stages",
        ])
    return PipelineRuntimeConfig.from_config(cfg)


class FixedStage:
    def __init__(self, result: Any) -> None:
        self.result, self.calls = result, 0
    def process(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.result
    def unload(self) -> None:
        pass


class KnownReID(PlayerReIDPredictor):
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint, self.calls = checkpoint, 0
        self.module = cast(Any, SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(num_slots=4)), matching_threshold=torch.tensor(.5)))

    def predict(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        b, v, _, p = values["human_kp"].shape[:4]
        valid = values["human_vis"].any(-1).any(2)
        return {"track_embedding": torch.eye(p)[None, None].expand(b, v, p, p).masked_fill(~valid[..., None], 0),
                "track_valid": valid, "is_player_logit": torch.full((b, v, p), 8.)}

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> PersonReIDResult:
        self.calls += 1
        return self.predict_observations(request, policy=policy)


class KnownSide:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint, self.calls = checkpoint, 0

    def process_observations(self, request: PersonObservationRequest, *, policy: PersonInferencePolicy) -> CourtSideResult:
        self.calls += 1
        sides = torch.tensor([False, False, True])
        return CourtSideResult(torch.where(sides, 8., -8.), sides)


def inputs(*, empty: bool = False) -> tuple[CourtKPResult, ObjectObservations, BallDetectionResult]:
    cameras = (_camera("cam0", [-8., -16., 9.]), _camera("cam1", [8., -16., 9.]), _camera("cam2", [5., 16., 9.]))
    local = (cameras[0], cameras[1], cameras[2].half_turned(True))
    template = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14]
    frames = 24
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
    balls = BallDetectionResult(ball_px / np.array([1280, 720], np.float32), ball_px, ball_visible, score)
    return court, people, balls


def setup_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, empty: bool = False) -> tuple[TennisSceneOrchestrator, tuple[Path, ...], list[Any]]:
    cfg = runtime(tmp_path)
    data = inputs(empty=empty)
    stages = [FixedStage(x) for x in data]
    plcs = KnownReID(cfg.plcs_reid_checkpoint)
    side = KnownSide(cfg.court_side_checkpoint)
    pipeline = TennisSceneOrchestrator(cfg, court=cast(Any, stages[0]), people=cast(Any, stages[1]), ball=cast(Any, stages[2]), plcs=cast(Any, plcs), side=cast(Any, side), body=None)
    paths = tuple(tmp_path / f"cam{i}.mp4" for i in range(3))
    for path in paths:
        path.write_bytes(b"video fixture")
    monkeypatch.setattr("src.tennis_scene.pipeline.orchestrator.probe_video_info", lambda _: VideoInfo(30., 1280, 720, 24))
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Automatic pipeline attempted human interaction")
    monkeypatch.setattr("builtins.input", forbidden)
    monkeypatch.setattr(cv2, "namedWindow", forbidden)
    return pipeline, paths, [*stages, plcs, side]


def test_headless_clip_once_and_v2_archive_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    assert scene.schema_version == 2 and scene.num_frames == 24 and scene.fps == 30.
    assert scene.metadata["court_reference"]["view_half_turns"] == [False, False, True]
    assert scene.player_kp_3d_vis is not None and scene.player_kp_3d_vis.all()
    assert scene.ball_3d_valid is not None and scene.ball_3d_valid.all()
    assert "blcs_association" not in scene.metadata["stage_status"]
    assert "blcs_association" not in pipeline.publication_identity()["checkpoints"]
    assert scene.player_valid is not None and not scene.player_valid.any()
    assert [stage.calls for stage in stages] == [1, 1, 1, 1, 1]
    output = tmp_path / "scene.npz"
    save_scene_result(scene, output)
    restored = load_scene_result(output)
    np.testing.assert_array_equal(restored.ball_3d_valid, scene.ball_3d_valid)
    np.testing.assert_allclose(restored.ball_3d, scene.ball_3d)
    pipeline.config = replace(pipeline.config, cache_source="load")
    again = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    np.testing.assert_array_equal(again.ball_3d, scene.ball_3d)
    assert [stage.calls for stage in stages] == [1, 1, 1, 1, 1]


def test_zero_detections_skip_player_association(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch, empty=True)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    assert scene.metadata["status"] == "empty"
    assert scene.metadata["court_reference"] is None
    assert scene.player_position.shape == (0, 24, 3)
    assert scene.ball_3d_valid is not None and not scene.ball_3d_valid.any()
    assert stages[-1].calls == stages[-2].calls == 0
    save_scene_result(scene, tmp_path / "empty.npz")
    assert load_scene_result(tmp_path / "empty.npz").schema_version == 2


def test_v2_missing_mask_rejected_before_any_archive_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, _ = setup_pipeline(tmp_path, monkeypatch, empty=True)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    scene.ball_3d_valid = None
    with pytest.raises(ValueError, match="ball_3d_valid"):
        save_scene_result(scene, tmp_path / "bad.npz")
    assert not (tmp_path / "bad.npz").exists()


def test_single_ball_missing_views_remain_invalid_without_identity_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    detected = stages[2].result
    detected.visibility[1:, 7] = False
    detected.score[1:, 7] = 0
    detected.ball_uv[1:, 7] = 0
    detected.ball_uv_px[1:, 7] = 0
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    assert scene.ball_3d_valid is not None
    assert not scene.ball_3d_valid[7]
    assert scene.ball_3d_valid.sum() == 23
    assert not scene.ball_3d[7].any()
    assert "blcs_association" not in scene.metadata["artifacts"]


def test_ball_boundary_rejects_multiple_detections() -> None:
    from src.tennis_scene.pipeline.components.ball_reconstruction import (
        single_ball_observations,
    )

    _, people, _ = inputs()
    ball = ObjectObservations(people.camera_ids, people.size, people.fps,
        np.repeat(people.uv_px[..., :1, :], 2, axis=2),
        np.repeat(people.confidence[..., :1], 2, axis=2),
        np.repeat(people.observed, 2, axis=2), np.zeros((3, 2), np.int64))
    with pytest.raises(ValueError, match="at most one"):
        single_ball_observations(ball, threshold=.5)


@pytest.mark.parametrize("missing_hips", [False, True])
def test_complete_body_path_corrects_yaw_from_coco17_and_keeps_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_hips: bool) -> None:
    from scipy.spatial.transform import Rotation

    from src.tennis_scene.pipeline.model_io.body import BodyGeometry, BodyParameters
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    raw = stages[1].result
    if missing_hips:
        confidence = raw.confidence.copy()
        confidence[..., 11:13] = 0
        raw = replace(raw, confidence=confidence)
    boxes = np.zeros((*raw.observed.shape, 3), np.float32)
    boxes[..., :2] = raw.uv_px.mean(-2)
    boxes[..., 2] = 100.
    stages[1].result = replace(raw, boxes_xys=boxes)
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
    config = replace(pipeline.config, enabled={**pipeline.config.enabled, "gvhmr": True}, processing_settings={**pipeline.config.processing_settings, "gvhmr": {"enabled": True}})
    pipeline = TennisSceneOrchestrator(config, court=cast(Any, stages[0]), people=cast(Any, stages[1]), ball=cast(Any, stages[2]), plcs=cast(Any, stages[3]), side=cast(Any, stages[4]), body=body)
    scene = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    assert scene.player_valid is not None and scene.player_valid.all()
    assert scene.player_heading_valid is not None and scene.player_heading_valid.all()
    assert scene.player_smpl_valid is not None and scene.player_smpl_valid.all()
    assert body.calls == 1 and body.unloaded
    assert scene.player_kp_3d is not None
    expected = np.column_stack((np.full(24, np.linspace(-.2, .2, 17)[11:13].mean()), -4 + np.arange(24)*.01,
                                np.full(24, np.linspace(.5, 1.8, 17)[11:13].mean()))) + [.03, 0., -.05]
    np.testing.assert_allclose(scene.player_position[0], expected, atol=1e-4)
    np.testing.assert_allclose(scene.player_yaw[0], 0, atol=1e-4)
    np.testing.assert_array_equal(scene.smpl_body_pose, 0)
    assert scene.metadata["body_placement"]["placement"] == "coco17_temporal_position_yaw_v1"
    assert scene.metadata["body_placement"]["players"]["0"]["scale"] == pytest.approx(1, abs=1e-5)
    if missing_hips:
        assert not scene.player_kp_3d_vis[..., 11:13].any()
    save_scene_result(scene, tmp_path / "body.npz")
    restored = load_scene_result(tmp_path / "body.npz")
    np.testing.assert_array_equal(restored.player_position, scene.player_position)
    pipeline.config = replace(config, cache_source="load")
    again = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
    np.testing.assert_array_equal(again.player_position, scene.player_position)
    assert body.calls == 1
    settings = dict(config.processing_settings)
    player_settings = dict(cast(dict[str, Any], settings["player_reconstruction"]))
    player_settings["placement"] = {**player_settings["placement"], "temporal_weight": .2}
    settings["player_reconstruction"] = player_settings
    pipeline.config = replace(pipeline.config, player_placement=replace(config.player_placement, temporal_weight=.2), processing_settings=settings)
    with pytest.raises(ValueError, match="[Ss]tale|identity"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"))
