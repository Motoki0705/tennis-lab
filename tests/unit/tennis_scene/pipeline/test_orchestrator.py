"""Tests for tennis_scene pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import src.tennis_scene.pipeline.orchestrator as orchestrator_module
from src.tennis_scene.pipeline.components.gvhmr import GVHMRResult
from src.tennis_scene.pipeline.dependency_graph import ResolutionResult, Stage
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator


def _make_orchestrator(tmp_path: Path) -> TennisSceneOrchestrator:
    resolution = ResolutionResult(
        enabled_order=(Stage.COURT_KP, Stage.GVHMR, Stage.PLCS),
        enabled_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        requested_set=frozenset({Stage.COURT_KP, Stage.GVHMR, Stage.PLCS}),
        disabled_reasons={},
    )
    return TennisSceneOrchestrator(
        court_kp_module=cast(Any, object()),
        gvhmr_config={
            "gvhmr_checkpoint": "gvhmr.ckpt",
            "detector": "dino",
            "yolo_checkpoint": "yolo.pt",
            "dino_checkpoint": "dino.pth",
            "dino_confidence": 0.35,
            "vitpose_checkpoint": "vitpose.pth",
            "hmr2_checkpoint": "hmr2.ckpt",
            "smplx_body_model_path": None,
            "track_selection": "auto",
            "num_tracks": 2,
            "save_result": True,
            "output_path": tmp_path / "gvhmr_result.json",
            "load_path": [tmp_path / "cam0.json", tmp_path / "cam1.json"],
            "device": "cpu",
        },
        player_association_module=cast(Any, object()),
        ball_detection_module=None,
        plcs_module=cast(Any, object()),
        blcs_module=None,
        resolution=resolution,
        device="cpu",
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
        def __init__(self, config: Any) -> None:
            self.config = config

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
    assert config.gvhmr_checkpoint == "gvhmr.ckpt"
    assert config.detector == "dino"
    assert config.dino_checkpoint == "dino.pth"
    assert config.dino_confidence == 0.35
    assert config.track_selection == "auto"
    assert config.save_result is True
    assert config.load_path == tmp_path / "cam1.json"
    assert config.output_path == tmp_path / "gvhmr_result_cam1.json"
