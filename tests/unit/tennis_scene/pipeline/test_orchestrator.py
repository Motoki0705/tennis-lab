"""Dependency order and synchronization contracts, independent of model execution."""

from pathlib import Path

import pytest

from src.tennis_scene.pipeline.dependency_graph import (
    Stage,
    build_default_dependency_graph,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.utils.video import VideoInfo


def flags() -> dict[str, bool]:
    return {key: True for key in ("court_kp", "person_observations", "ball_detection", "plcs_association", "camera_geometry", "player_reconstruction", "ball_reconstruction", "gvhmr")}


def test_player_association_supplies_side_and_ball_uses_detector() -> None:
    enabled = flags()
    result = build_default_dependency_graph(enabled).resolve_from_enabled(enabled)
    order = list(result.enabled_order)
    assert order.index(Stage.PLCS_ASSOCIATION) < order.index(Stage.CAMERA_GEOMETRY)
    assert order.index(Stage.BALL_DETECTION) < order.index(Stage.BALL_RECONSTRUCTION)
    assert order.index(Stage.CAMERA_GEOMETRY) < order.index(Stage.GVHMR)


def test_ball_only_pipeline_and_strict_missing_dependencies() -> None:
    enabled = flags()
    for key in ("player_reconstruction", "gvhmr"):
        enabled[key] = False
    result = build_default_dependency_graph(enabled).resolve_from_enabled(enabled)
    assert Stage.PLCS_ASSOCIATION in result.enabled_set
    assert Stage.CAMERA_GEOMETRY in result.enabled_set
    enabled["ball_detection"] = False
    with pytest.raises(ValueError, match="missing dependency"):
        build_default_dependency_graph(enabled).resolve_from_enabled(enabled)


@pytest.mark.parametrize("different", [VideoInfo(25., 1920, 1080, 20), VideoInfo(30., 1280, 720, 20), VideoInfo(30., 1920, 1080, 19)])
def test_unsynchronized_sources_fail_before_inference(monkeypatch: pytest.MonkeyPatch, different: VideoInfo) -> None:
    def probe(path: Path) -> VideoInfo:
        return VideoInfo(30., 1920, 1080, 20) if path.name == "a.mp4" else different
    monkeypatch.setattr("src.tennis_scene.pipeline.orchestrator.probe_video_info", probe)
    instance = object.__new__(TennisSceneOrchestrator)
    with pytest.raises(ValueError, match="synchronized"):
        instance._probe_synced_video_infos([Path("a.mp4"), Path("b.mp4")], max_frames=None)
