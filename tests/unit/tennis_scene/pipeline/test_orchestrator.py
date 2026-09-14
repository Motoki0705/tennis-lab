"""Tests for tennis_scene pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import src.tennis_scene.pipeline.orchestrator as orchestrator_module
from src.tennis_scene.motion_alignment.similarity import SimilarityConfig
from src.tennis_scene.pipeline.components.motion_alignment import (
    MotionAlignmentModule,
    PlayerMotionConfig,
)
from src.tennis_scene.pipeline.components.player_association import (
    PlayerAssociationApplied,
    PlayerAssociationResult,
    PlayerAssociationSegment,
)
from src.tennis_scene.pipeline.components.plcs import PLCSResult
from src.tennis_scene.pipeline.dependency_graph import ResolutionResult, Stage
from src.tennis_scene.pipeline.model_io.gvhmr import GVHMRResult
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.configuration import PathRole
from src.utils.video import VideoInfo
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_gvhmr_config,
    make_resolver,
)


def _make_orchestrator(tmp_path: Path) -> TennisSceneOrchestrator:
    resolution = ResolutionResult(
        enabled_order=(Stage.COURT_KP, Stage.GVHMR, Stage.PLCS),
        enabled_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        requested_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        disabled_reasons={},
    )
    return TennisSceneOrchestrator(
        court_kp_module=cast(Any, object()),
        gvhmr_config=make_gvhmr_config(
            tmp_path,
            source="load",
            save_result=True,
            output_path=tmp_path / "gvhmr_result.json",
            load_path=tmp_path / "gvhmr_result.json",
        ),
        gvhmr_chain=None,
        player_association_module=cast(Any, object()),
        motion_alignment_module=cast(Any, object()),
        ball_detection_module=None,
        plcs_module=cast(Any, object()),
        blcs_module=None,
        resolution=resolution,
        device="cpu",
        resolver=make_resolver(tmp_path),
    )


def test_run_gvhmr_invokes_module_in_process_with_camera_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, Path, int | None]] = []
    expected = GVHMRResult(
        smpl_body_pose=np.zeros((1, 2, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((1, 2, 3), dtype=np.float32),
        smpl_betas=np.zeros((1, 10), dtype=np.float32),
        smpl_vertices_local=None,
        human_kp_2d=np.zeros((1, 2, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((1, 2, 17), dtype=np.float32),
        bbx_xys=np.zeros((1, 2, 3), dtype=np.float32),
        track_ids=np.array([3], dtype=np.int32),
    )

    class FakeGVHMRModule:
        def __init__(self, config: Any, chain: Any) -> None:
            self.config = config
            assert chain is None

        def process(self, video_path: str | Path, max_frames: int | None = None) -> Any:
            calls.append((self.config, Path(video_path), max_frames))
            return expected

    monkeypatch.setattr(orchestrator_module, "GVHMRModule", FakeGVHMRModule)

    orchestrator = _make_orchestrator(tmp_path)
    result = orchestrator._run_gvhmr(
        Path("cam1.mp4"),
        camera_index=1,
        num_cameras=2,
        max_frames=2,
    )

    assert result is expected
    assert len(calls) == 1
    config, video_path, max_frames = calls[0]
    assert video_path == Path("cam1.mp4")
    assert max_frames == 2
    assert config.gvhmr_checkpoint == (tmp_path / "ckpt/gvhmr.ckpt").resolve()
    assert config.detector == "dino"
    assert config.dino_checkpoint == (tmp_path / "ckpt/dino.pth").resolve()
    assert config.runtime.dino_detector.confidence == 0.35
    assert config.track_selection == "auto"
    assert config.save_result is True
    assert config.load_path == tmp_path / "gvhmr_result_cam1.json"
    assert config.output_path == tmp_path / "gvhmr_result_cam1.json"


class _FakeCourtKPModule:
    def __init__(self, result: Any) -> None:
        self.result = result

    def process(self, *args: Any, **kwargs: Any) -> Any:
        return self.result


class _FakePLCSModule:
    def __init__(self, result: PLCSResult) -> None:
        self.result = result

    def process(self, *args: Any, **kwargs: Any) -> PLCSResult:
        return self.result


def test_run_places_plcs_motion_unchanged_and_records_the_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default source keeps the PLCS arrays bit-for-bit and labels them."""
    num_frames = 3
    resolution = ResolutionResult(
        enabled_order=(Stage.COURT_KP, Stage.GVHMR, Stage.PLCS),
        enabled_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        requested_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        disabled_reasons={},
    )
    monkeypatch.setattr(
        orchestrator_module,
        "probe_video_info",
        lambda _: VideoInfo(fps=30.0, width=640, height=360, frame_count=num_frames),
    )
    court_result = SimpleNamespace(
        keypoints=np.zeros((1, num_frames, 20, 2), dtype=np.float32),
        visibility=np.ones((1, num_frames, 20), dtype=np.float32),
        frame_indices=np.array([0], dtype=np.int64),
    )
    position: np.ndarray = np.arange(
        num_frames * 3, dtype=np.float32
    ).reshape(1, num_frames, 3)
    yaw = np.linspace(0.0, 0.5, num_frames, dtype=np.float32).reshape(1, num_frames)
    plcs_result = PLCSResult(
        position=position,
        yaw=yaw,
        track_ids=np.array([5], dtype=np.int32),
    )
    aligned = PlayerAssociationApplied(
        human_kp_2d=np.zeros((1, 1, num_frames, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((1, 1, num_frames, 17), dtype=np.float32),
        smpl_body_pose=np.full((1, num_frames, 63), 0.5, dtype=np.float32),
        smpl_global_orient=np.full((1, num_frames, 3), 0.25, dtype=np.float32),
        smpl_betas=np.full((1, 10), 0.125, dtype=np.float32),
        smpl_vertices_local=np.full(
            (1, num_frames, 4, 3), 0.75, dtype=np.float32
        ),
        track_ids=np.array([5], dtype=np.int32),
        track_ids_by_camera=[np.array([5], dtype=np.int32)],
    )
    association = PlayerAssociationResult(
        camera_ids=["cam0"],
        canonical_player_ids=np.array([5], dtype=np.int32),
        segments=[
            PlayerAssociationSegment(
                start_frame=0,
                end_frame=num_frames,
                assignments=np.zeros((1, 1), dtype=np.int32),
            )
        ],
        reference_camera="cam0",
    )
    orchestrator = TennisSceneOrchestrator(
        court_kp_module=cast(Any, _FakeCourtKPModule(court_result)),
        gvhmr_config=make_gvhmr_config(
            tmp_path,
            source="load",
            save_result=False,
            load_path=tmp_path / "gvhmr_result.json",
        ),
        gvhmr_chain=None,
        player_association_module=cast(Any, object()),
        motion_alignment_module=MotionAlignmentModule(
            PlayerMotionConfig(
                source="plcs",
                scale_mode="fixed",
                smpl_joint_regressor=tmp_path / "unused_regressor.pt",
                similarity=SimilarityConfig(),
            )
        ),
        ball_detection_module=None,
        plcs_module=cast(Any, _FakePLCSModule(plcs_result)),
        blcs_module=None,
        resolution=resolution,
        device="cpu",
        resolver=make_resolver(tmp_path),
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_gvhmr_multicamera",
        lambda **_: (association, aligned),
    )
    video_path = tmp_path / "data/cam0.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    result = orchestrator.run(
        [video_path],
        video_role=PathRole.DATA,
        max_frames=None,
        frame_index=0,
        camera_ids=["cam0"],
    )

    assert result.player_position is position
    assert result.player_yaw is yaw
    assert result.smpl_body_pose is aligned.smpl_body_pose
    assert result.smpl_global_orient is aligned.smpl_global_orient
    assert result.smpl_vertices_local is aligned.smpl_vertices_local
    assert result.metadata["player_motion"] == {"source": "plcs"}
    assert result.metadata["track_ids"] == [5]
