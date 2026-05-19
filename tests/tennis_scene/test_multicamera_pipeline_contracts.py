from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionConfig,
    BallDetectionModule,
    BallDetectionResult,
)
from src.tennis_scene.pipeline.components.blcs import BLCSConfig, BLCSModule
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationResult,
    PlayerAssociationSegment,
    apply_player_association,
)
from src.tennis_scene.pipeline.components.plcs import PLCSConfig, PLCSModule
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.video.types import VideoInfo


class _DummyPLCSPredictor:
    def predict(
        self,
        *,
        human_kp: torch.Tensor,
        court_kp: torch.Tensor,
        human_vis: torch.Tensor | None,
        human_mask: torch.Tensor,
        court_vis: torch.Tensor | None,
        denormalize: bool,
    ) -> dict[str, torch.Tensor]:
        assert human_kp.shape == (2, 3, 4, 17, 2)
        assert court_kp.shape == (2, 3, 4, 14, 2)
        assert human_vis is not None and human_vis.shape == (2, 3, 4, 17)
        assert human_mask.shape == (2, 3, 4)
        assert court_vis is not None and court_vis.shape == (2, 3, 4, 14)
        assert denormalize
        return {
            "position_meters": torch.zeros((2, 4, 3), dtype=torch.float32),
            "yaw_radians": torch.zeros((2, 4), dtype=torch.float32),
        }


class _DummyBLCSPredictor:
    def predict(
        self,
        *,
        ball_uv: torch.Tensor,
        court_kp: torch.Tensor,
        ball_vis: torch.Tensor,
        ball_mask: torch.Tensor,
        court_vis: torch.Tensor | None,
        denormalize: bool,
    ) -> dict[str, torch.Tensor]:
        assert ball_uv.shape == (1, 3, 4, 2)
        assert court_kp.shape == (1, 3, 4, 14, 2)
        assert ball_vis.shape == (1, 3, 4)
        assert ball_mask.shape == (1, 3, 4)
        assert court_vis is not None and court_vis.shape == (1, 3, 4, 14)
        assert denormalize
        return {"position": torch.ones((1, 4, 3), dtype=torch.float32)}


def _make_gvhmr_result(camera_offset: float = 0.0) -> GVHMRResult:
    p, t = 2, 4
    human_kp_2d = np.zeros((p, t, 17, 2), dtype=np.float32)
    for player_index in range(p):
        for frame_index in range(t):
            human_kp_2d[player_index, frame_index, :, 0] = (
                100 * player_index + frame_index + camera_offset
            )
            human_kp_2d[player_index, frame_index, :, 1] = (
                200 * player_index + frame_index + camera_offset
            )
    smpl_body_pose = np.zeros((p, t, 63), dtype=np.float32)
    smpl_global_orient = np.zeros((p, t, 3), dtype=np.float32)
    for player_index in range(p):
        smpl_body_pose[player_index] = player_index + 10
        smpl_global_orient[player_index] = player_index + 20
    return GVHMRResult(
        smpl_body_pose=smpl_body_pose,
        smpl_global_orient=smpl_global_orient,
        smpl_betas=np.stack(
            [
                np.full((10,), 30, dtype=np.float32),
                np.full((10,), 31, dtype=np.float32),
            ],
            axis=0,
        ),
        smpl_vertices_local=None,
        human_kp_2d=human_kp_2d,
        human_kp_vis=np.ones((p, t, 17), dtype=np.float32),
        bbx_xys=np.zeros((p, t, 3), dtype=np.float32),
        track_ids=np.array([100, 200], dtype=np.int32),
    )


def test_court_kp_result_requires_multicamera_shape() -> None:
    result = CourtKPResult(
        keypoints=np.zeros((3, 4, 14, 2), dtype=np.float32),
        visibility=np.ones((3, 4, 14), dtype=np.float32),
        frame_indices=np.arange(4, dtype=np.int32),
    )
    assert result.validate(num_keypoints=14) == (True, [])

    legacy_result = CourtKPResult(
        keypoints=np.zeros((4, 14, 2), dtype=np.float32),
        visibility=np.ones((4, 14), dtype=np.float32),
        frame_indices=np.arange(4, dtype=np.int32),
    )
    valid, errors = legacy_result.validate(num_keypoints=14)
    assert not valid
    assert any("(N, T, 14, 2)" in error for error in errors)


def test_ball_detection_result_requires_multicamera_shape() -> None:
    result = BallDetectionResult(
        ball_uv=np.zeros((3, 4, 2), dtype=np.float32),
        ball_uv_px=np.zeros((3, 4, 2), dtype=np.float32),
        visibility=np.ones((3, 4), dtype=np.bool_),
        score=np.ones((3, 4), dtype=np.float32),
    )
    assert result.validate() == (True, [])

    legacy_result = BallDetectionResult(
        ball_uv=np.zeros((4, 2), dtype=np.float32),
        ball_uv_px=np.zeros((4, 2), dtype=np.float32),
        visibility=np.ones((4,), dtype=np.bool_),
        score=np.ones((4,), dtype=np.float32),
    )
    valid, errors = legacy_result.validate()
    assert not valid
    assert any("(N, T, 2)" in error for error in errors)


def test_player_association_applies_cross_camera_and_temporal_id_switch() -> None:
    association = PlayerAssociationResult(
        camera_ids=["cam0", "cam1"],
        canonical_player_ids=np.array([0, 1], dtype=np.int32),
        reference_camera="cam0",
        segments=[
            PlayerAssociationSegment(
                start_frame=0,
                end_frame=2,
                assignments=np.array([[0, 1], [1, 0]], dtype=np.int32),
            ),
            PlayerAssociationSegment(
                start_frame=2,
                end_frame=4,
                assignments=np.array([[1, 1], [0, 0]], dtype=np.int32),
            ),
        ],
    )
    valid, errors = association.validate(
        num_frames=4,
        local_player_counts=[2, 2],
    )
    assert valid, errors

    applied = apply_player_association(
        gvhmr_results=[_make_gvhmr_result(0.0), _make_gvhmr_result(10.0)],
        video_infos=[
            VideoInfo(fps=30.0, width=1000, height=1000, frame_count=4),
            VideoInfo(fps=30.0, width=1000, height=1000, frame_count=4),
        ],
        association=association,
    )

    assert applied.human_kp_2d.shape == (2, 2, 4, 17, 2)
    # canonical player 0 uses cam0 local 0 before the split, then local 1.
    assert applied.human_kp_2d[0, 0, 1, 0, 0] == pytest.approx(1 / 1000)
    assert applied.human_kp_2d[0, 0, 2, 0, 0] == pytest.approx(102 / 1000)
    # canonical player 0 uses cam1 local 1 throughout.
    assert applied.human_kp_2d[0, 1, 0, 0, 0] == pytest.approx(110 / 1000)
    assert applied.smpl_body_pose[0, 1, 0] == pytest.approx(10)
    assert applied.smpl_body_pose[0, 2, 0] == pytest.approx(11)
    assert applied.track_ids.tolist() == [0, 1]
    assert [ids.tolist() for ids in applied.track_ids_by_camera] == [
        [100, 200],
        [100, 200],
    ]


def test_player_association_rejects_duplicate_local_assignment() -> None:
    association = PlayerAssociationResult(
        camera_ids=["cam0"],
        canonical_player_ids=np.array([0, 1], dtype=np.int32),
        reference_camera="cam0",
        segments=[
            PlayerAssociationSegment(
                start_frame=0,
                end_frame=4,
                assignments=np.array([[0], [0]], dtype=np.int32),
            )
        ],
    )

    valid, errors = association.validate(
        num_frames=4,
        local_player_counts=[2],
    )

    assert not valid
    assert any("same local player" in error for error in errors)


def test_plcs_wrapper_passes_multicamera_sequence_to_predictor() -> None:
    module = PLCSModule(
        PLCSConfig(
            checkpoint_path="dummy.ckpt",
            device="cpu",
        )
    )
    module._predictor = _DummyPLCSPredictor()

    result = module.process(
        human_kp_2d=np.zeros((2, 3, 4, 17, 2), dtype=np.float32),
        court_kp=np.zeros((3, 4, 14, 2), dtype=np.float32),
        human_kp_vis=np.ones((2, 3, 4, 17), dtype=np.float32),
        court_vis=np.ones((3, 4, 14), dtype=np.float32),
    )

    assert result.position.shape == (2, 4, 3)
    assert result.yaw.shape == (2, 4)
    assert result.track_ids is not None
    assert result.track_ids.tolist() == [0, 1]


def test_plcs_wrapper_rejects_single_camera_sequence_shape() -> None:
    module = PLCSModule(PLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = _DummyPLCSPredictor()

    with pytest.raises(ValueError, match=r"\(P, N, T, 17, 2\)"):
        module.process(
            human_kp_2d=np.zeros((2, 4, 17, 2), dtype=np.float32),
            court_kp=np.zeros((4, 14, 2), dtype=np.float32),
        )


def test_blcs_wrapper_passes_multicamera_sequence_to_predictor() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = _DummyBLCSPredictor()

    result = module.process(
        ball_uv=np.zeros((3, 4, 2), dtype=np.float32),
        court_kp=np.zeros((3, 4, 14, 2), dtype=np.float32),
        ball_vis=np.array(
            [
                [True, False, False, True],
                [False, False, True, False],
                [False, False, False, False],
            ],
            dtype=np.bool_,
        ),
        court_vis=np.ones((3, 4, 14), dtype=np.float32),
    )

    assert result.ball_3d.shape == (4, 3)
    assert result.visibility is not None
    assert result.visibility.tolist() == [True, False, True, True]
    assert np.all(result.ball_3d[1] == 0.0)


def test_blcs_wrapper_rejects_single_camera_sequence_shape() -> None:
    module = BLCSModule(BLCSConfig(checkpoint_path="dummy.ckpt", device="cpu"))
    module._predictor = _DummyBLCSPredictor()

    with pytest.raises(ValueError, match=r"\(N, T, 2\)"):
        module.process(
            ball_uv=np.zeros((4, 2), dtype=np.float32),
            court_kp=np.zeros((4, 14, 2), dtype=np.float32),
        )


def test_ball_detection_module_stacks_per_camera_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    module = BallDetectionModule(
        BallDetectionConfig(
            checkpoint="dummy.ckpt",
            device="cpu",
            score_threshold=0.5,
        )
    )
    module._pipeline = object()

    def fake_predict_video(
        video_path: str | Path,
        *,
        max_frames: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        index = int(Path(video_path).stem[-1])
        coords = np.full((4, 2), 0.1 * (index + 1), dtype=np.float32)
        score = np.ones((4,), dtype=np.float32)
        return coords, score

    monkeypatch.setattr(module, "_predict_video", fake_predict_video)
    monkeypatch.setattr(
        "src.tennis_scene.pipeline.components.ball_detection.probe_video_info",
        lambda _path: VideoInfo(fps=30.0, width=100, height=50, frame_count=4),
    )

    result = module.process([Path("cam0.mp4"), Path("cam1.mp4")], max_frames=4)

    assert result.ball_uv.shape == (2, 4, 2)
    assert result.ball_uv_px.shape == (2, 4, 2)
    assert result.visibility.shape == (2, 4)
    assert np.allclose(result.ball_uv[1], 0.2)


def test_orchestrator_rejects_unsynchronized_video_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = TennisSceneOrchestrator.__new__(TennisSceneOrchestrator)
    infos = {
        Path("cam0.mp4"): VideoInfo(fps=30.0, width=1920, height=1080, frame_count=4),
        Path("cam1.mp4"): VideoInfo(fps=30.0, width=1920, height=1080, frame_count=5),
    }
    monkeypatch.setattr(
        "src.tennis_scene.pipeline.orchestrator.probe_video_info",
        lambda path: infos[Path(path)],
    )

    with pytest.raises(ValueError, match="synchronized"):
        orchestrator._probe_synced_video_infos(
            [Path("cam0.mp4"), Path("cam1.mp4")],
            max_frames=None,
        )
