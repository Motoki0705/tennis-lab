"""End-of-run diagnostics must distinguish observations from reconstruction."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import fields
from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tennis_scene import archive
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.scripts import run_pipeline
from src.tennis_scene.scripts.run_pipeline import log_pipeline_summary


def _scene() -> SceneResult:
    return SceneResult(num_frames=4, fps=60.0, width=640, height=360,
        court_kp=np.zeros((3, 4, 14, 2), np.float32), court_vis=np.zeros((3, 4, 14), np.float32),
        player_position=np.zeros((0, 4, 3), np.float32), player_yaw=np.zeros((0, 4), np.float32),
        ball_vis=np.array([[True, True, True, False], [False, True, True, False], [False, False, False, False]]),
        ball_3d_valid=np.array([False, True, False, False]),
        # All coordinates are zero: coordinate values must not determine validity.
        ball_3d=np.zeros((4, 3), np.float32))


@pytest.mark.parametrize(("case", "observed", "valid"), [
    ("mixed", "3/4", "1/4"),
    ("zero", "0/4", "0/4"),
    ("no_2d", "unavailable", "1/4"),
    ("no_3d", "3/4", "unavailable"),
    ("neither", "unavailable", "unavailable"),
])
def test_ball_summary_counts_actual_masks_without_changing_scene(
    caplog: pytest.LogCaptureFixture, case: str, observed: str, valid: str,
) -> None:
    scene = _scene()
    if case == "zero":
        scene.ball_vis = np.zeros((3, 4), bool)
        scene.ball_3d_valid = np.zeros(4, bool)
    if case in {"no_2d", "neither"}:
        scene.ball_vis = None
    if case in {"no_3d", "neither"}:
        scene.ball_3d_valid = None
    if case == "no_3d":
        scene.ball_3d = None  # 2D diagnostics do not require 3D coordinates.
    original = deepcopy(scene)

    with caplog.at_level(logging.INFO, logger="src.tennis_scene.scripts.run_pipeline"):
        log_pipeline_summary(scene)

    assert f"Ball 2D observed frames (>=1 view): {observed}" in caplog.text
    assert f"Ball 3D valid frames: {valid}" in caplog.text
    for field in fields(scene):
        actual, expected = getattr(scene, field.name), getattr(original, field.name)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_pipeline_entrypoint_logs_both_counts_after_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    scene = _scene()
    video = tmp_path / "input.mp4"
    video.touch()
    runtime = SimpleNamespace(video_paths=(video,), camera_ids=("cam0",), device="cpu", max_frames=None,
                              output_path=tmp_path / "scene.npz")
    monkeypatch.setattr(PipelineRuntimeConfig, "from_config", lambda cfg: runtime)
    monkeypatch.setattr(TennisSceneOrchestrator, "from_runtime_config", lambda config: SimpleNamespace(run=lambda **kwargs: scene))
    saved = []

    def save(result: SceneResult, path: Path) -> None:
        saved.append((result, path))
        assert "Pipeline Summary:" not in caplog.text

    monkeypatch.setattr(archive, "save_scene_result", save)
    with caplog.at_level(logging.INFO, logger="src.tennis_scene.scripts.run_pipeline"):
        assert unwrap(run_pipeline.main)(OmegaConf.create()) == 0
    assert len(saved) == 1 and saved[0][0] is scene and saved[0][1] == runtime.output_path
    assert "Ball 2D observed frames (>=1 view): 3/4" in caplog.text
    assert "Ball 3D valid frames: 1/4" in caplog.text
