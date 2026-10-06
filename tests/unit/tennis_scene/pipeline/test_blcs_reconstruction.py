"""BLCS input gauge, source-time restoration and observation-backed validity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.blcs.inference.predictor import BLCSPredictor
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.blcs import (
    BallReconstructionInput,
    BLCSReconstructionModule,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryResult
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.geometry.triangulation import PinholeCamera, PointRejection
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    CourtConfig,
    court_keypoints_3d,
)


def camera(name: str, center: list[float]) -> PinholeCamera:
    c = np.asarray(center, np.float64)
    forward = np.array([0., 0., 1.]) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    r = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]]), r, -r @ c)


def trajectory(times: np.ndarray) -> NDArray[np.float32]:
    result: NDArray[np.float32] = np.column_stack((1 + .5 * times, -8 + 2 * times, np.full(len(times), 1.4))).astype(np.float32)
    return result


class KnownPredictor:
    input_profile = "multiview"

    def __init__(self, *, invalid: bool = False) -> None:
        self.cursor = 0
        self.calls: list[dict[str, Any]] = []
        self.invalid = invalid

    def predict_multiview_arrays(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        frames = kwargs["ball_uv"].shape[1]
        positions = trajectory(np.arange(self.cursor, self.cursor + frames) / 30.)
        self.cursor += frames
        if self.invalid:
            positions[0] = np.nan
        return SimpleNamespace(position=torch.from_numpy(positions).unsqueeze(0))


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, frames: int = 24, fps: float = 30.,
            invalid: bool = False) -> tuple[BLCSReconstructionModule, BallReconstructionInput, KnownPredictor, np.ndarray]:
    ids = ("cam0", "cam1", "cam2")
    cameras = (camera(ids[0], [-8., -16., 9.]), camera(ids[1], [8., -16., 9.]), camera(ids[2], [5., 16., 9.]))
    source = ClipSource("fixture", tuple(SourceVideo(c, tmp_path / f"{c}.mp4", "fixture", frames, fps, 1280, 720) for c in ids))
    xyz = trajectory(np.arange(frames) / fps)
    uv = np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)
    objects = ObjectObservations(ids, source.size, fps, uv[:, :, None, None],
        np.ones((3, frames, 1, 1), np.float32), np.ones((3, frames, 1), bool), np.zeros((3, 1), np.int64))
    observations = single_ball_observations(objects, threshold=0.)
    template = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14]
    local = (cameras[0], cameras[1], cameras[2].half_turned(True))
    local_uv = np.stack([c.project(template)[0] for c in local]) / [1279, 719]
    court = CourtKPResult(np.repeat(local_uv[:, None], frames, axis=1).astype(np.float32),
        np.repeat(np.broadcast_to(np.linspace(.1, 1, 14, dtype=np.float32), (3, 14))[:, None], frames, axis=1),
        np.arange(frames, dtype=np.int32), {"output_keypoint_contract": "camera_view_v2"})
    geometry = CameraGeometryResult(ids, ids[0], (False, False, True), cameras, cast(Any, None), {})
    request = BallReconstructionInput(source, CameraAlignmentOutput(geometry), observations, court, ids)
    roots = RuntimePathRoots.from_mapping({"project_root": str(tmp_path), "data_root": "data", "checkpoint_root": "ckpt",
        "artifact_root": "outputs", "output_root": "outputs", "cache_root": ".cache", "external_asset_root": "third_party"}, repository_root=tmp_path)
    module = BLCSReconstructionModule(ids, checkpoint=tmp_path / "ckpt/model.ckpt", resolver=PathResolver(roots),
        device="cpu", window_size=64, reprojection_px=20., min_frames=5)
    predictor = KnownPredictor(invalid=invalid)
    module._model_fps = 30.
    monkeypatch.setattr(module, "_load_predictor", lambda: cast(BLCSPredictor, predictor))
    return module, request, predictor, xyz


def test_camera_local_court_is_half_turned_and_normalized_like_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, predictor, xyz = fixture(tmp_path, monkeypatch)
    output = module.process(request)
    assert output.ball is not None
    np.testing.assert_allclose(output.ball.trajectory.positions, xyz, atol=1e-5)
    assert output.ball.trajectory.valid.all()
    assert request.alignment.geometry is not None
    template = court_keypoints_3d(CourtConfig(.914, None)).numpy()[:14]
    expected = np.stack([c.project(template)[0] / [1280, 720] for c in request.alignment.geometry.cameras])
    np.testing.assert_allclose(predictor.calls[0]["court_kp"], expected, atol=1e-6)
    np.testing.assert_allclose(predictor.calls[0]["court_vis"][2], request.court.visibility[2, 0, list(CAMERA_VIEW_HALF_TURN_INDEX)])
    assert predictor.calls[0]["denormalize"] is True
    assert predictor.calls[0]["court_keypoint_document"]["court_keypoints"]["contract_id"] == "physical_courtkp20_v1"


def test_network_predictions_do_not_restore_missing_observation_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, _, _ = fixture(tmp_path, monkeypatch)
    request.observations.visibility[:, :, 8:11] = False
    request.observations.visibility[:, 1:, 15] = False
    output = module.process(request)
    assert output.ball is not None
    invalid = [8, 9, 10, 15]
    assert not output.ball.trajectory.valid[invalid].any()
    assert not output.ball.trajectory.inliers[:, invalid].any()
    assert (output.ball.trajectory.positions[invalid] == 0).all()
    assert (output.ball.trajectory.reasons[invalid] == PointRejection.INSUFFICIENT_VIEWS).all()


def test_windowing_and_checkpoint_fps_restore_every_source_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, predictor, xyz = fixture(tmp_path, monkeypatch, frames=270, fps=60.)
    output = module.process(request)
    assert output.ball is not None
    assert [call["ball_uv"].shape[1] for call in predictor.calls] == [64, 64, 8]
    assert output.ball.trajectory.positions.shape == (270, 3)
    np.testing.assert_allclose(output.ball.trajectory.positions, xyz, atol=1e-5)
    assert output.ball.trajectory.valid.all()


def test_reprojection_failure_rejects_prediction_instead_of_triangulation_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, _, _ = fixture(tmp_path, monkeypatch)
    request.observations.uv_px[:, :, 7] += 200
    output = module.process(request)
    assert output.ball is not None
    assert not output.ball.trajectory.valid[7]
    assert output.ball.trajectory.reasons[7] == PointRejection.REPROJECTION
    assert (output.ball.trajectory.positions[7] == 0).all()


def test_nonfinite_model_output_fails_explicitly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, _, _ = fixture(tmp_path, monkeypatch, invalid=True)
    with pytest.raises(RuntimeError, match="invalid trajectory"):
        module.process(request)


def test_insufficient_support_avoids_model_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module, request, predictor, _ = fixture(tmp_path, monkeypatch)
    request.observations.visibility[:] = False
    output = module.process(request)
    assert output.ball is not None and output.ball.status == "ball_insufficient_support"
    assert not predictor.calls
