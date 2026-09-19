"""CPU contracts for audited detector-produced ball observations."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from src.tennis_scene.dataset_pipeline import build, detector_ball
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.tennis_scene.dataset_pipeline.detector_ball import (
    DetectorBallSettings,
    observe_detector_ball,
)
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionResult
from src.utils.checksum import FileIntegrityError, dual_sha256
from src.utils.video import VideoInfo


@pytest.fixture
def recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(
        config_dir=str(root / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="build_broadcast_slcs_dataset")
    cfg.court_calibration_clips = None
    cfg.excluded_clips = {}
    cfg.dataset_clip_ids = None
    cfg.clip_ids = None
    cfg.paths.checkpoint_root = str(tmp_path)
    monkeypatch.setattr(
        "src.tennis_scene.dataset_pipeline.configuration.load_dataset_manifest",
        lambda path: SimpleNamespace(
            clips={"video/clip": SimpleNamespace(video_id="video")}
        ),
    )
    ball = OmegaConf.load(
        root / "src/tennis_scene/configs/pipeline.yaml"
    ).ball_detection
    for key in ("enabled", "source", "save_result", "output_path", "load_path"):
        del ball[key]
    checkpoint = tmp_path / "ball.ckpt"
    checkpoint.write_bytes(b"model bytes")
    ball.checkpoint = checkpoint.name
    ball.checkpoint_sha256 = dual_sha256(checkpoint)
    with open_dict(cfg):
        cfg.ball_source = "detector"
        cfg.ball_detector = ball
    return cfg


@pytest.fixture
def clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ClipManifest:
    source = tmp_path / "source"
    source.mkdir()
    raw = dict(
        version=2,
        dataset_id="test",
        clip_id="video/clip",
        video_id="video",
        clip_name="clip",
        fps=30.0,
        num_frames=3,
        width=100,
        height=80,
        global_start_sec=0.0,
        global_end_sec=0.1,
        camera_ids=["cam1", "cam0"],
        video_paths=["cam1.mp4", "cam0.mp4"],
        cameras=[],
        sync_source="test",
        exported_at="test",
    )
    (source / "clip.json").write_text(json.dumps(raw))
    for name in ("cam1.mp4", "cam0.mp4"):
        (source / name).write_bytes(name.encode())
    monkeypatch.setattr(
        detector_ball, "probe_video_info", lambda path: VideoInfo(30.0, 100, 80, 3)
    )
    return ClipManifest.load(source)


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    def create(config):
        def process(videos, **kwargs):
            assert [path.name for path in videos] == ["cam1.mp4", "cam0.mp4"]
            result = BallDetectionResult(
                np.full((2, 3, 2), 0.5, np.float32),
                np.tile(np.array([49.5, 39.5], np.float32), (2, 3, 1)),
                np.ones((2, 3), bool),
                np.full((2, 3), 0.8, np.float32),
            )
            result.save(config.output_path)
            return result

        return SimpleNamespace(process=process)

    mock = MagicMock(side_effect=create)
    monkeypatch.setattr(detector_ball, "BallDetectionModule", mock)
    return mock


def settings(recipe: DictConfig) -> DetectorBallSettings:
    value = DatasetBuildConfig.from_config(recipe).ball_detector
    assert value is not None
    return value


def test_cold_warm_and_camera_order(recipe, clip, model, tmp_path):
    output = tmp_path / "observations"
    runtime = DatasetBuildConfig.from_config(recipe)
    build._import_ball(runtime, clip, output)
    build._import_ball(runtime, clip, output)
    assert model.call_count == 1
    config = model.call_args.args[0]
    assert config.source == "execute" and config.save_result
    assert config.output_path.parent != output
    receipt = json.loads((output / "ball_import.metadata.json").read_text())
    assert receipt["camera_ids"] == ["cam1", "cam0"]
    assert receipt["is_ground_truth"] is False
    assert receipt["settings"]["trajectory_gate"]["enabled"] is True
    assert receipt["checkpoint_sha256"] == recipe.ball_detector.checkpoint_sha256
    assert not (clip.clip_dir / "observations").exists()


@pytest.mark.parametrize(
    "change",
    [
        "media",
        "checkpoint",
        "settings",
        "manifest",
        "result",
        "receipt",
        "missing_result",
        "missing_receipt",
    ],
)
def test_changed_or_partial_cache_rejected(recipe, clip, model, tmp_path, change):
    output = tmp_path / "observations"
    selected = settings(recipe)
    observe_detector_ball(clip, output, selected)
    if change == "media":
        clip.media_path("cam0").write_bytes(b"changed")
    elif change == "checkpoint":
        selected.config.checkpoint.write_bytes(b"changed")
    elif change == "settings":
        selected = replace(
            selected, config=replace(selected.config, score_threshold=0.7)
        )
    elif change == "manifest":
        clip.manifest_path.write_text(clip.manifest_path.read_text() + " ")
    elif change in {"result", "receipt"}:
        name = (
            "ball_detection_result.json"
            if change == "result"
            else "ball_import.metadata.json"
        )
        (output / name).write_text("{}")
    else:
        name = (
            "ball_detection_result.json"
            if change == "missing_result"
            else "ball_import.metadata.json"
        )
        (output / name).unlink()
    with pytest.raises(FileIntegrityError if change == "checkpoint" else ValueError):
        observe_detector_ball(clip, output, selected)
    assert model.call_count == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("ball_uv", float("nan")),
        ("ball_uv", 1.1),
        ("score", 1.1),
        ("visibility", 2),
        ("ball_uv_px", 4.0),
    ],
)
def test_corrupt_result_rejected_even_with_matching_digest(
    recipe, clip, model, tmp_path, field, value
):
    output = tmp_path / "observations"
    observe_detector_ball(clip, output, settings(recipe))
    path = output / "ball_detection_result.json"
    raw = json.loads(path.read_text())
    if field in {"ball_uv", "ball_uv_px"}:
        raw[field][0][0][0] = value
    else:
        raw[field][0][0] = value
    path.write_text(json.dumps(raw))
    receipt_path = output / "ball_import.metadata.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["result_sha256"] = dual_sha256(path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        observe_detector_ball(clip, output, settings(recipe))
    assert model.call_count == 1


@pytest.mark.parametrize(
    "info",
    [
        VideoInfo(29.0, 100, 80, 3),
        VideoInfo(30.0, 99, 80, 3),
        VideoInfo(30.0, 100, 80, 4),
    ],
)
def test_media_contract_rejected_before_model(
    recipe, clip, model, tmp_path, monkeypatch, info
):
    monkeypatch.setattr(detector_ball, "probe_video_info", lambda path: info)
    with pytest.raises(ValueError, match="shape/FPS"):
        observe_detector_ball(clip, tmp_path / "observations", settings(recipe))
    model.assert_not_called()


@pytest.mark.parametrize("shape", [(1, 3, 2), (2, 2, 2)])
def test_model_shape_mismatch_never_published(recipe, clip, model, tmp_path, shape):
    def create(config):
        def process(*args, **kwargs):
            result = BallDetectionResult(
                np.zeros(shape, np.float32),
                np.zeros(shape, np.float32),
                np.zeros(shape[:2], bool),
                np.zeros(shape[:2], np.float32),
            )
            result.save(config.output_path)
            return result

        return SimpleNamespace(process=process)

    model.side_effect = create
    output = tmp_path / "observations"
    with pytest.raises(ValueError, match="camera/frame"):
        observe_detector_ball(clip, output, settings(recipe))
    assert not (output / "ball_detection_result.json").exists()
    assert not (output / "ball_import.metadata.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("score_threshold", float("nan")),
        ("source", "load"),
        ("save_result", False),
        ("checkpoint_sha256", "invalid"),
        ("unknown", 1),
    ],
)
def test_invalid_detector_config(recipe, field, value):
    with open_dict(recipe.ball_detector):
        recipe.ball_detector[field] = value
    with pytest.raises((ValueError, TypeError)):
        DatasetBuildConfig.from_config(recipe)


@pytest.mark.parametrize("source", ["outsource", "saved_scene"])
def test_old_sources_forbid_detector_settings_but_remain_valid(recipe, source):
    recipe.ball_source = source
    with pytest.raises(ValueError, match="required only"):
        DatasetBuildConfig.from_config(recipe)
    with open_dict(recipe):
        del recipe.ball_detector
    assert DatasetBuildConfig.from_config(recipe).ball_detector is None


def test_detector_requires_settings(recipe):
    with open_dict(recipe):
        del recipe.ball_detector
    with pytest.raises(ValueError, match="required only"):
        DatasetBuildConfig.from_config(recipe)


def test_detector_rejects_outsource_court_crop(recipe):
    recipe.court.ball_crop_margins = {"cam0": 0.2}
    with pytest.raises(ValueError, match="null court.ball_crop_margins"):
        DatasetBuildConfig.from_config(recipe)


def test_detector_court_cache_uses_run_receipt(
    recipe, clip, model, tmp_path, monkeypatch
):
    output = tmp_path / "observations"
    runtime = DatasetBuildConfig.from_config(recipe)
    runtime = replace(
        runtime,
        court=replace(runtime.court, checkpoint=settings(recipe).config.checkpoint),
    )
    build._import_ball(runtime, clip, output)
    observe = MagicMock(return_value=(np.zeros((2, 3, 14, 2)), np.zeros((2, 3, 3)), {}))
    monkeypatch.setattr(build, "observe_static_court", observe)
    monkeypatch.setattr(build.torch.cuda, "empty_cache", lambda: None)
    build._court(recipe, runtime, clip, output)
    build._court(recipe, runtime, clip, output)
    assert observe.call_count == 1
    receipt = json.loads((output / "court.json").read_text())
    assert receipt["identity"]["ball_annotation_sha256"] == {
        "detector": dual_sha256(output / "ball_import.metadata.json")
    }


@pytest.mark.parametrize("asset", ["checkpoint", "media"])
def test_input_changed_during_inference_never_published(
    recipe, clip, model, tmp_path, asset
):
    selected = settings(recipe)
    create = model.side_effect

    def changing_model(config):
        instance = create(config)
        process = instance.process

        def changed(*args, **kwargs):
            result = process(*args, **kwargs)
            path = (
                selected.config.checkpoint
                if asset == "checkpoint"
                else clip.media_path("cam0")
            )
            path.write_bytes(b"changed during execution")
            return result

        return SimpleNamespace(process=changed)

    model.side_effect = changing_model
    output = tmp_path / "observations"
    with pytest.raises(FileIntegrityError if asset == "checkpoint" else ValueError):
        observe_detector_ball(clip, output, selected)
    assert not (output / "ball_import.metadata.json").exists()
    assert not (output / "ball_detection_result.json").exists()


def test_missing_saved_output_is_not_success(recipe, clip, model, tmp_path):
    model.side_effect = lambda config: SimpleNamespace(
        process=lambda *args, **kwargs: None
    )
    output = tmp_path / "observations"
    with pytest.raises(FileNotFoundError):
        observe_detector_ball(clip, output, settings(recipe))
    assert not (output / "ball_import.metadata.json").exists()


def test_outsource_import_dispatch_unchanged(
    recipe, clip, model, tmp_path, monkeypatch
):
    runtime = replace(
        DatasetBuildConfig.from_config(recipe),
        ball_source="outsource",
        ball_detector=None,
    )
    imported = MagicMock()
    monkeypatch.setattr(build, "import_ball", imported)
    build._import_ball(runtime, clip, tmp_path)
    imported.assert_called_once_with(clip.clip_dir, tmp_path)
    model.assert_not_called()


def test_saved_scene_import_preserves_bytes(recipe, clip, model, tmp_path):
    generated = tmp_path / "generated"
    observe_detector_ball(clip, generated, settings(recipe))
    source = clip.clip_dir / "observations"
    source.mkdir()
    for name in ("ball_detection_result.json", "ball_import.metadata.json"):
        (source / name).write_bytes((generated / name).read_bytes())
    runtime = replace(
        DatasetBuildConfig.from_config(recipe),
        ball_source="saved_scene",
        ball_detector=None,
    )
    output = tmp_path / "imported"
    output.mkdir()
    build._import_ball(runtime, clip, output)
    build._import_ball(runtime, clip, output)
    assert (source / "ball_detection_result.json").read_bytes() == (
        output / "ball_detection_result.json"
    ).read_bytes()
    assert model.call_count == 1
