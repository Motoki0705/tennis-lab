"""Boundary tests for the composed tennis-scene GVHMR model chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import src.submodules.models as submodule_models
import src.tennis_scene.pipeline.model_io.gvhmr as gvhmr_io
from src.tennis_scene.pipeline.model_io.gvhmr import (
    GVHMRChainAdapter,
    GVHMRChainRequest,
    GVHMRContractError,
    GVHMRResult,
    build_gvhmr_chain,
)
from src.utils.video import VideoInfo
from tests.unit.tennis_scene.pipeline.config_factories import make_gvhmr_config


class _PredictorSpy:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[object] = []
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def predict(self, request: object) -> object:
        self.requests.append(request)
        return self.result


class _VertexSpy:
    def __init__(self, *, num_frames: int, num_vertices: int = 8) -> None:
        self.num_frames = num_frames
        self.num_vertices = num_vertices
        self.requests: list[dict[str, torch.Tensor]] = []

    def reconstruct(self, request: dict[str, torch.Tensor]) -> torch.Tensor:
        self.requests.append(request)
        return torch.ones(
            (self.num_frames, self.num_vertices, 3), dtype=torch.float32
        )


def _smpl_parameters(num_frames: int) -> dict[str, torch.Tensor]:
    return {
        "body_pose": torch.zeros((num_frames, 63), dtype=torch.float32),
        "betas": torch.zeros((num_frames, 10), dtype=torch.float32),
        "global_orient": torch.zeros((num_frames, 3), dtype=torch.float32),
        "transl": torch.zeros((num_frames, 3), dtype=torch.float32),
    }


def _make_adapter(
    *,
    num_frames: int = 4,
    track_result: object | None = None,
    raw_track: torch.Tensor | None = None,
    keypoints: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    mesh_result: submodule_models.GvhmrResult | None = None,
) -> tuple[
    GVHMRChainAdapter,
    _PredictorSpy,
    _PredictorSpy,
    _PredictorSpy,
    _PredictorSpy,
    _VertexSpy,
]:
    if raw_track is None:
        raw_track = torch.tensor(
            [[10.0, 20.0, 30.0, 60.0]] * num_frames,
            dtype=torch.float32,
        )
    if keypoints is None:
        keypoints = torch.zeros((num_frames, 17, 3), dtype=torch.float32)
        keypoints[..., 2] = 1.0
    if features is None:
        features = torch.ones((num_frames, 1024), dtype=torch.float32)
    if mesh_result is None:
        mesh_result = submodule_models.GvhmrResult(
            smpl_params_incam=_smpl_parameters(num_frames),
            smpl_params_global=_smpl_parameters(num_frames),
            K_fullimg=torch.eye(3, dtype=torch.float32).expand(
                num_frames, -1, -1
            ),
        )

    if track_result is None:
        track_result = submodule_models.TrackResult(
            tracks={7: raw_track}, num_frames=num_frames
        )
    tracker = _PredictorSpy(track_result)
    pose = _PredictorSpy(submodule_models.Pose2DResult(keypoints=keypoints))
    feature = _PredictorSpy(submodule_models.ImageFeatureResult(features=features))
    mesh = _PredictorSpy(mesh_result)
    vertices = _VertexSpy(num_frames=num_frames)
    adapter = GVHMRChainAdapter(
        tracker=cast(Any, tracker),
        pose_model=cast(Any, pose),
        feature_model=cast(Any, feature),
        mesh_model=cast(Any, mesh),
        vertex_reconstructor=cast(Any, vertices),
    )
    return adapter, tracker, pose, feature, mesh, vertices


def _request(video_path: Path, *, max_frames: int | None = None) -> GVHMRChainRequest:
    return GVHMRChainRequest(
        video_path=video_path,
        max_frames=max_frames,
        num_tracks=1,
        interactive=False,
        bbox_enlarge=1.2,
        static_cam=True,
    )


@pytest.fixture
def video_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "input.mp4"
    path.touch()
    monkeypatch.setattr(
        gvhmr_io,
        "probe_video_info",
        lambda _: VideoInfo(fps=30.0, width=1920, height=1080, frame_count=4),
    )
    return path


def test_decoded_result_roundtrip_preserves_optional_vertices(tmp_path: Path) -> None:
    result = GVHMRResult(
        smpl_body_pose=np.zeros((1, 2, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 2, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        smpl_vertices_local=None,
        human_kp_2d=np.zeros((1, 2, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((1, 2, 17), dtype=np.float32),
        bbx_xys=np.ones((1, 2, 3), dtype=np.float32),
        track_ids=np.array([7], dtype=np.int32),
    )
    path = tmp_path / "gvhmr.json"

    result.save(path)
    loaded = GVHMRResult.load(path)

    assert loaded.smpl_vertices_local is None
    np.testing.assert_array_equal(loaded.track_ids, result.track_ids)


def test_decoded_result_rejects_missing_required_field() -> None:
    with pytest.raises(GVHMRContractError, match="missing required fields.*track_ids"):
        GVHMRResult.from_dict(
            {
                "smpl_body_pose": [],
                "smpl_global_orient": [],
                "smpl_betas": [],
                "human_kp_2d": [],
                "human_kp_vis": [],
                "bbx_xys": [],
            }
        )


def test_valid_chain_executes_typed_requests_and_decodes_result(
    video_path: Path,
) -> None:
    adapter, tracker, pose, feature, mesh, vertices = _make_adapter()

    result = adapter.predict(_request(video_path, max_frames=4))

    assert isinstance(result, GVHMRResult)
    assert result.smpl_body_pose.shape == (1, 4, 63)
    assert result.smpl_vertices_local is not None
    assert result.smpl_vertices_local.shape == (1, 4, 8, 3)
    np.testing.assert_array_equal(result.track_ids, np.array([7], dtype=np.int32))
    assert isinstance(tracker.requests[0], submodule_models.TrackRequest)
    assert isinstance(pose.requests[0], submodule_models.Pose2DRequest)
    assert isinstance(feature.requests[0], submodule_models.ImageFeatureRequest)
    assert isinstance(mesh.requests[0], submodule_models.GvhmrRequest)
    mesh_request = mesh.requests[0]
    assert mesh_request.kp2d.shape == (4, 17, 3)
    assert mesh_request.bbx_xys.shape == (4, 3)
    assert mesh_request.f_imgseq.shape == (4, 1024)
    assert len(vertices.requests) == 1


def test_rank_one_keypoints_stop_before_feature_and_mesh_entry(
    video_path: Path,
) -> None:
    adapter, _, _, feature, mesh, _ = _make_adapter(keypoints=torch.ones(4))

    with pytest.raises(GVHMRContractError, match="keypoints must have shape"):
        adapter.predict(_request(video_path))

    assert feature.requests == []
    assert mesh.requests == []


def test_wrong_feature_shape_stops_before_mesh_entry(video_path: Path) -> None:
    adapter, _, _, _, mesh, _ = _make_adapter(
        features=torch.ones((4, 512), dtype=torch.float32)
    )

    with pytest.raises(GVHMRContractError, match="features must have shape"):
        adapter.predict(_request(video_path))

    assert mesh.requests == []


def test_wrong_keypoint_dtype_stops_before_feature_entry(video_path: Path) -> None:
    keypoints = torch.zeros((4, 17, 3), dtype=torch.float64)
    adapter, _, _, feature, mesh, _ = _make_adapter(keypoints=keypoints)

    with pytest.raises(TypeError, match="keypoints must have dtype torch.float32"):
        adapter.predict(_request(video_path))

    assert feature.requests == []
    assert mesh.requests == []


def test_mismatched_feature_device_stops_before_mesh_entry(video_path: Path) -> None:
    adapter, _, _, _, mesh, _ = _make_adapter(
        features=torch.empty((4, 1024), dtype=torch.float32, device="meta")
    )

    with pytest.raises(GVHMRContractError, match="must be on cpu"):
        adapter.predict(_request(video_path))

    assert mesh.requests == []


def test_track_sequence_must_match_video_metadata(video_path: Path) -> None:
    adapter, _, pose, _, _, _ = _make_adapter(num_frames=3)

    with pytest.raises(GVHMRContractError, match="frame count must match"):
        adapter.predict(_request(video_path))

    assert pose.requests == []


def test_wrong_tracker_result_type_stops_all_downstream_models(
    video_path: Path,
) -> None:
    adapter, tracker, pose, feature, mesh, vertices = _make_adapter(
        track_result={"tracks": {}}
    )

    with pytest.raises(TypeError, match="tracker must return TrackResult"):
        adapter.predict(_request(video_path))

    assert len(tracker.requests) == 1
    assert pose.requests == []
    assert feature.requests == []
    assert mesh.requests == []
    assert vertices.requests == []


def test_semantically_invalid_track_stops_before_pose_entry(video_path: Path) -> None:
    invalid_track = torch.tensor(
        [[30.0, 20.0, 10.0, 60.0]] * 4,
        dtype=torch.float32,
    )
    adapter, _, pose, feature, mesh, _ = _make_adapter(raw_track=invalid_track)

    with pytest.raises(GVHMRContractError, match="positive-area"):
        adapter.predict(_request(video_path))

    assert pose.requests == []
    assert feature.requests == []
    assert mesh.requests == []


def test_missing_smpl_key_stops_before_vertex_model_entry(video_path: Path) -> None:
    incam = _smpl_parameters(4)
    incam.pop("body_pose")
    mesh_result = submodule_models.GvhmrResult(
        smpl_params_incam=incam,
        smpl_params_global=_smpl_parameters(4),
        K_fullimg=torch.eye(3, dtype=torch.float32).expand(4, -1, -1),
    )
    adapter, _, _, _, _, vertices = _make_adapter(mesh_result=mesh_result)

    with pytest.raises(GVHMRContractError, match="missing required SMPL keys"):
        adapter.predict(_request(video_path))

    assert vertices.requests == []


def test_invalid_video_metadata_stops_before_tracker_entry(
    video_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, tracker, _, _, _, _ = _make_adapter()
    monkeypatch.setattr(
        gvhmr_io,
        "probe_video_info",
        lambda _: VideoInfo(fps=30.0, width=0, height=1080, frame_count=4),
    )

    with pytest.raises(GVHMRContractError, match="width must be a positive"):
        adapter.predict(_request(video_path))

    assert tracker.requests == []


def test_missing_video_stops_before_tracker_entry(tmp_path: Path) -> None:
    adapter, tracker, _, _, _, _ = _make_adapter()

    with pytest.raises(FileNotFoundError, match="input video not found"):
        adapter.predict(_request(tmp_path / "missing.mp4"))

    assert tracker.requests == []


def test_factory_selects_dino_once_and_constructs_resolved_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constructed: dict[str, list[dict[str, Any]]] = {}

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            constructed.setdefault(type(self).__name__, []).append(kwargs)

    class FakeDinoTracker(FakeModel):
        pass

    class FakeYoloTracker(FakeModel):
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            raise AssertionError("YOLO must not be selected in DINO mode")

    monkeypatch.setattr(submodule_models, "DinoPersonTracker", FakeDinoTracker)
    monkeypatch.setattr(submodule_models, "YoloPersonTracker", FakeYoloTracker)
    monkeypatch.setattr(submodule_models, "ViTPosePose2D", FakeModel)
    monkeypatch.setattr(submodule_models, "Hmr2FeatureExtractor", FakeModel)
    monkeypatch.setattr(submodule_models, "GvhmrMeshRecovery", FakeModel)
    monkeypatch.setattr(submodule_models, "SmplVertexReconstructor", FakeModel)

    chain = build_gvhmr_chain(
        make_gvhmr_config(tmp_path, detector="dino", dino_confidence=0.42)
    )

    assert isinstance(chain, GVHMRChainAdapter)
    assert constructed["FakeDinoTracker"] == [
        {
            "checkpoint": (tmp_path / "ckpt/dino.pth").resolve(),
            "repository": (tmp_path / "third_party/DINO").resolve(),
            "confidence": 0.42,
            "short_side": 800,
            "max_long_side": 1333,
            "device": "cpu",
        }
    ]
