"""Dependency order and synchronization contracts, independent of model execution."""

from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.pipeline.feature_flags import validate_requested_features
from src.tennis_scene.pipeline.source import build_clip_source
from src.utils.video import VideoInfo


def flags() -> dict[str, bool]:
    from src.tennis_scene.pipeline.feature_flags import OPTIONAL_FEATURES
    return dict.fromkeys(OPTIONAL_FEATURES, True)


def _runtime_with_assets(tmp_path: Path) -> Any:
    from src.tennis_scene.pipeline.definition import enabled_model_assets
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
        materialize_assets,
        runtime,
    )
    cfg = runtime(tmp_path)
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path)
    return cfg


def _source(tmp_path: Path) -> Any:
    from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
    return ClipSource("clip", tuple(SourceVideo(c, tmp_path / f"{c}.mp4", "hash", 24, 30., 1280, 720) for c in ("cam0", "cam1", "cam2")))


def test_runtime_dependencies_come_from_component_declarations(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline.definition import standard_definition
    from src.tennis_scene.pipeline.runner import ComponentRunner
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore
    source = _source(tmp_path)
    runner = ComponentRunner(standard_definition(_runtime_with_assets(tmp_path), source, code_identity="test"), ClipStore(tmp_path / "store", {"clip": "clip"}))
    order = runner.order
    assert order.index("court_calibration") < order.index("player_selection/cam0")
    assert order.index("person_tracking/cam0") < order.index("player_selection/cam0") < order.index("pose_estimation/cam0")
    nodes = {n.name: n for n in standard_definition(_runtime_with_assets(tmp_path), source, code_identity="test")}
    assert nodes["person_detection/cam0"].bindings == {}
    assert nodes["player_selection/cam0"].settings["rule"]["max_candidates"] == 6
    assert nodes["player_selection/cam0"].settings["rule"]["min_presence_fraction"] == .25
    assert order.index("court_side") < order.index("player_association") < order.index("player_triangulation")
    assert order.index("person_tracking/cam2") < order.index("player_association")
    assert order.index("ball_detection/cam0") < order.index("ball_refiner_2d/cam0") < order.index("ball_points/cam0") < order.index("court_side")
    for consumer in ("court_side", "camera_alignment", "ball_triangulation"):
        assert nodes[consumer].bindings["ball_cam0"] == "ball_points/cam0"
        assert nodes[consumer].io.inputs["ball_cam0"].schema == "ball_points"
        assert nodes[consumer].assembler.ball_threshold == 0
    assert order.index("court_side") < order.index("camera_alignment")
    assert order.index("body_view_selection") < order.index("gvhmr") < order.index("body_placement")


def test_missing_enabled_asset_stops_the_definition(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.definition import standard_definition
    cfg = _runtime_with_assets(tmp_path)
    cfg.people.vitpose_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="vitpose"):
        standard_definition(cfg, _source(tmp_path), code_identity="test")
    # A disabled feature neither needs nor records its assets.
    disabled = replace(cfg, enabled={**cfg.enabled, "person_observations": False, "player_reconstruction": False, "gvhmr": False})
    standard_definition(disabled, _source(tmp_path), code_identity="test")


def test_missing_coco_checkpoint_does_not_select_available_player_weights(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline.definition import standard_definition

    cfg = _runtime_with_assets(tmp_path)
    legacy = cfg.roots.checkpoint_root / "player_detection/chat-player-v1-e8-best-pr937.pth"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"available legacy checkpoint")
    cfg.people.detector_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint0029_4scale_swin"):
        standard_definition(cfg, _source(tmp_path), code_identity="test")


def test_the_association_records_its_encoder_weights_and_needs_them_only_with_people(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.definition import standard_definition
    source = _source(tmp_path)
    cfg = _runtime_with_assets(tmp_path)
    assert cfg.association_encoder_weights is not None and cfg.association_encoder_weights.is_relative_to(tmp_path)
    node = next(node for node in standard_definition(cfg, source, code_identity="test") if node.name == "player_association")
    assert node.source == "execute" and node.settings["assets"]["encoder"]["path"] == str(cfg.association_encoder_weights)
    assert node.settings["config"]["players_per_side"] == 1
    cfg.association_encoder_weights.unlink()
    with pytest.raises(FileNotFoundError, match="person_vit_clip_reid"):
        standard_definition(cfg, source, code_identity="test")
    # Geometry-only association still needs CLIP for the default tracker.
    geometry_only = replace(cfg, player_association=replace(cfg.player_association, appearance=None), association_encoder_weights=None)
    with pytest.raises(FileNotFoundError, match="person_vit_clip_reid"):
        standard_definition(geometry_only, source, code_identity="test")
    no_people = replace(cfg, enabled={**cfg.enabled, "person_observations": False, "player_reconstruction": False, "gvhmr": False})
    node = next(node for node in standard_definition(no_people, source, code_identity="test") if node.name == "player_association")
    assert node.settings["assets"] == {"enabled": False}


def test_a_side_without_ball_detection_fails_when_the_definition_is_built(tmp_path: Path) -> None:
    from dataclasses import replace

    from src.tennis_scene.pipeline.definition import standard_definition
    source = _source(tmp_path)
    cfg = _runtime_with_assets(tmp_path)
    # The side is decided from the ball alone: without a ball detector it can only be loaded.
    ballless = replace(cfg, enabled={**cfg.enabled, "ball_detection": False, "ball_reconstruction": False})
    with pytest.raises(ValueError, match="court_side decides sides from the ball alone"):
        standard_definition(ballless, source, code_identity="test")
    loaded = replace(ballless, component_sources={**cfg.component_sources, **dict.fromkeys(("court_side", "ball_detection", "ball_refiner_2d", "ball_points"), "load")})
    assert any(node.name == "court_side" and node.source == "load" for node in standard_definition(loaded, source, code_identity="test"))


def test_ball_only_features_and_strict_missing_dependencies() -> None:
    enabled = flags()
    for key in ("player_reconstruction", "gvhmr"):
        enabled[key] = False
    validate_requested_features(enabled)
    enabled["ball_detection"] = False
    with pytest.raises(ValueError, match="missing dependency"):
        validate_requested_features(enabled)
    with pytest.raises(ValueError, match="must be exactly"):
        validate_requested_features({**flags(), "court_side": True})


def test_every_standard_component_has_one_execution_mode_and_node(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline.contracts import STANDARD_COMPONENTS
    from src.tennis_scene.pipeline.definition import standard_definition
    cfg = _runtime_with_assets(tmp_path)
    assert tuple(cfg.component_sources) == STANDARD_COMPONENTS
    nodes = standard_definition(cfg, _source(tmp_path), code_identity="test")
    assert {node.name.split("/")[0] for node in nodes} == set(STANDARD_COMPONENTS)


@pytest.mark.parametrize("different", [VideoInfo(25., 1920, 1080, 20), VideoInfo(30., 1280, 720, 20), VideoInfo(30., 1920, 1080, 19)])
def test_unsynchronized_sources_fail_before_inference(monkeypatch: pytest.MonkeyPatch, different: VideoInfo) -> None:
    def probe(path: Path) -> VideoInfo:
        return VideoInfo(30., 1920, 1080, 20) if path.name == "a.mp4" else different
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info", probe)
    monkeypatch.setattr("src.tennis_scene.pipeline.source.dual_sha256", lambda _: "source_hash")
    with pytest.raises(ValueError, match="FPS|frame count"):
        build_clip_source([Path("a.mp4"), Path("b.mp4"), Path("c.mp4")], ["a", "b", "c"])
