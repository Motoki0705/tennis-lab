"""Dependency order and synchronization contracts, independent of model execution."""

from pathlib import Path

import pytest

from src.tennis_scene.pipeline.feature_flags import validate_requested_features
from src.tennis_scene.pipeline.source import build_clip_source
from src.utils.video import VideoInfo


def flags() -> dict[str, bool]:
    return {key: True for key in ("court_kp", "person_observations", "ball_detection", "camera_geometry", "player_reconstruction", "ball_reconstruction", "gvhmr")}


def test_runtime_dependencies_come_from_component_declarations(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
    from src.tennis_scene.pipeline.definition import standard_definition
    from src.tennis_scene.pipeline.runner import ComponentRunner
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import runtime
    source = ClipSource("clip", tuple(SourceVideo(c, tmp_path / f"{c}.mp4", "hash", 24, 30., 1280, 720) for c in ("cam0", "cam1", "cam2")))
    runner = ComponentRunner(standard_definition(runtime(tmp_path), source, code_identity="test"), ClipStore(tmp_path / "store", {"clip": "clip"}))
    order = runner.order
    assert order.index("court_calibration") < order.index("person_detection/cam0")
    assert order.index("player_association") < order.index("player_triangulation")
    assert order.index("court_side") < order.index("camera_alignment")
    assert order.index("body_view_selection") < order.index("gvhmr") < order.index("body_placement")


def test_executing_an_unimplemented_node_fails_when_the_definition_is_built(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
    from src.tennis_scene.pipeline.definition import standard_definition
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import runtime
    source = ClipSource("clip", tuple(SourceVideo(c, tmp_path / f"{c}.mp4", "hash", 24, 30., 1280, 720) for c in ("cam0", "cam1", "cam2")))
    cfg = runtime(tmp_path)
    for node in ("player_association", "court_side"):
        executed = replace(cfg, component_sources={**cfg.component_sources, node: "execute"})
        with pytest.raises(ValueError, match=f"{node} has no model implementation"):
            standard_definition(executed, source, code_identity="test")


def test_ball_only_features_and_strict_missing_dependencies() -> None:
    enabled = flags()
    for key in ("player_reconstruction", "gvhmr"):
        enabled[key] = False
    validate_requested_features(enabled)
    enabled["ball_detection"] = False
    with pytest.raises(ValueError, match="missing dependency"):
        validate_requested_features(enabled)


@pytest.mark.parametrize("different", [VideoInfo(25., 1920, 1080, 20), VideoInfo(30., 1280, 720, 20), VideoInfo(30., 1920, 1080, 19)])
def test_unsynchronized_sources_fail_before_inference(monkeypatch: pytest.MonkeyPatch, different: VideoInfo) -> None:
    def probe(path: Path) -> VideoInfo:
        return VideoInfo(30., 1920, 1080, 20) if path.name == "a.mp4" else different
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info", probe)
    monkeypatch.setattr("src.tennis_scene.pipeline.source.dual_sha256", lambda _: "source_hash")
    with pytest.raises(ValueError, match="FPS|frame count"):
        build_clip_source([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], ["a", "b", "c"])
