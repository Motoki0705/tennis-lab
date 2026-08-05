"""Tests for tennis_scene player association."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.tennis_scene.pipeline.components import player_association as module_under_test
from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationModule,
    PlayerAssociationResult,
    PlayerAssociationSegment,
)
from src.utils.video import VideoInfo
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_player_association_config,
)


def _make_gvhmr_result(*, num_players: int = 2, num_frames: int = 3) -> GVHMRResult:
    rng = np.random.default_rng(0)
    return GVHMRResult(
        smpl_body_pose=rng.normal(size=(num_players, num_frames, 63)).astype(
            np.float32
        ),
        smpl_global_orient=rng.normal(size=(num_players, num_frames, 3)).astype(
            np.float32
        ),
        smpl_betas=rng.normal(size=(num_players, 10)).astype(np.float32),
        smpl_vertices_local=rng.normal(size=(num_players, num_frames, 4, 3)).astype(
            np.float32
        ),
        human_kp_2d=rng.uniform(0, 640, size=(num_players, num_frames, 17, 2)).astype(
            np.float32
        ),
        human_kp_vis=np.ones((num_players, num_frames, 17), dtype=np.float32),
        bbx_xys=rng.normal(size=(num_players, num_frames, 3)).astype(np.float32),
        track_ids=np.array([11, 22], dtype=np.int32)[:num_players],
    )


def test_single_camera_process_creates_identity_association_without_ui(
    monkeypatch,
    caplog,
    tmp_path: Path,
) -> None:
    def _fail_named_window(*_args, **_kwargs) -> None:
        raise AssertionError("manual UI must not be opened for a single camera")

    monkeypatch.setattr(module_under_test.cv2, "namedWindow", _fail_named_window)
    association = PlayerAssociationModule(
        make_player_association_config(tmp_path, reference_camera="cam0")
    )

    with caplog.at_level(logging.INFO):
        result = association.process(
            gvhmr_results=[_make_gvhmr_result(num_players=2, num_frames=3)],
            video_paths=[Path("cam0.mp4")],
            video_infos=[VideoInfo(fps=30.0, width=640, height=360, frame_count=3)],
            camera_ids=["cam0"],
        )

    assert result.camera_ids == ["cam0"]
    assert result.reference_camera == "cam0"
    np.testing.assert_array_equal(result.canonical_player_ids, np.array([0, 1]))
    assert len(result.segments) == 1
    segment = result.segments[0]
    assert segment.start_frame == 0
    assert segment.end_frame == 3
    np.testing.assert_array_equal(
        segment.assignments,
        np.array([[0], [1]], dtype=np.int32),
    )
    assert "single camera -> identity association" in caplog.text


def test_single_camera_identity_apply_preserves_reference_player_order(
    tmp_path: Path,
) -> None:
    gvhmr = _make_gvhmr_result(num_players=2, num_frames=3)
    association = PlayerAssociationModule(
        make_player_association_config(tmp_path, reference_camera=0)
    )
    result = association.process(
        gvhmr_results=[gvhmr],
        video_paths=[Path("cam0.mp4")],
        video_infos=[VideoInfo(fps=30.0, width=640, height=360, frame_count=3)],
        camera_ids=["cam0"],
    )

    applied = association.apply(
        gvhmr_results=[gvhmr],
        video_infos=[VideoInfo(fps=30.0, width=640, height=360, frame_count=3)],
        association=result,
    )

    assert applied.human_kp_2d.shape == (2, 1, 3, 17, 2)
    np.testing.assert_allclose(applied.human_kp_2d[:, 0, :, :, 0], gvhmr.human_kp_2d[..., 0] / 640.0)
    np.testing.assert_allclose(applied.human_kp_2d[:, 0, :, :, 1], gvhmr.human_kp_2d[..., 1] / 360.0)
    np.testing.assert_allclose(applied.smpl_body_pose, gvhmr.smpl_body_pose)
    np.testing.assert_array_equal(applied.track_ids, np.array([0, 1], dtype=np.int32))
    assert len(applied.track_ids_by_camera) == 1
    np.testing.assert_array_equal(applied.track_ids_by_camera[0], gvhmr.track_ids)


def test_multicamera_process_still_delegates_to_manual_ui(
    monkeypatch,
    tmp_path: Path,
) -> None:
    expected = PlayerAssociationResult(
        camera_ids=["cam0", "cam1"],
        canonical_player_ids=np.array([0], dtype=np.int32),
        segments=[
            PlayerAssociationSegment(
                start_frame=0,
                end_frame=3,
                assignments=np.array([[0, 0]], dtype=np.int32),
            )
        ],
        reference_camera="cam0",
    )
    calls: list[tuple[str, ...]] = []

    def _fake_manual_ui(self, **kwargs) -> PlayerAssociationResult:
        del self
        calls.append(tuple(kwargs["camera_ids"]))
        return expected

    monkeypatch.setattr(PlayerAssociationModule, "_process_manual_ui", _fake_manual_ui)
    association = PlayerAssociationModule(
        make_player_association_config(tmp_path, reference_camera="cam0")
    )

    result = association.process(
        gvhmr_results=[
            _make_gvhmr_result(num_players=1, num_frames=3),
            _make_gvhmr_result(num_players=1, num_frames=3),
        ],
        video_paths=[Path("cam0.mp4"), Path("cam1.mp4")],
        video_infos=[
            VideoInfo(fps=30.0, width=640, height=360, frame_count=3),
            VideoInfo(fps=30.0, width=640, height=360, frame_count=3),
        ],
        camera_ids=["cam0", "cam1"],
    )

    assert result is expected
    assert calls == [("cam0", "cam1")]
