"""Runtime-contract tests for ball visualization orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import av
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from PIL import Image

from src.tasks.ball_detection.configuration import validate_visualization
from src.tasks.ball_detection.scripts import visualize as visualize_script
from src.tasks.ball_detection.visualization import orchestrator
from src.tasks.ball_detection.visualization.orchestrator import (
    build_runtime_config,
    overlay_playback_fps,
)
from src.utils.configuration import ConfigurationError
from src.utils.device import DeviceSelectionError
from src.utils.paths import PROJECT_ROOT
from tests.support.tasks.ball_detection.store import frame, write_store_clip


def _visualization_config(*, device: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/ball_detection/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="visualize", overrides=[f"run.device={device}"])


def test_runtime_config_keeps_resolved_torch_device() -> None:
    config = _visualization_config(device="cpu")
    assert isinstance(config.data.image_size, ListConfig)
    assert isinstance(config.data.augmentation.normalize_imagenet.mean, ListConfig)
    assert isinstance(config.visualization.draw.gt_color_rgb, ListConfig)

    runtime = build_runtime_config(config)

    assert isinstance(runtime.device, torch.device)
    assert runtime.device == torch.device("cpu")
    assert runtime.image_size_hw == (288, 512)
    assert runtime.imagenet_mean == (0.485, 0.456, 0.406)
    assert runtime.draw.gt_color_rgb == (255, 96, 96)
    assert runtime.layout == "grid_2x2"
    assert runtime.grid.background_rgb == (18, 18, 18)
    assert runtime.overlay_output_width == 960
    assert runtime.start_frame == 0
    assert runtime.overlay_styles == ("ring",)


@pytest.mark.parametrize("layout", ["unknown", "", "GRID_2X2"])
def test_unknown_layout_is_rejected_at_runtime_and_cli_boundary(layout: str) -> None:
    config = _visualization_config(device="cpu")
    config.visualization.layout = layout
    with pytest.raises(ValueError, match="visualization.layout"):
        build_runtime_config(config)
    with pytest.raises(ConfigurationError, match="visualization.layout"):
        validate_visualization(config)


def test_layout_is_required() -> None:
    config = _visualization_config(device="cpu")
    with open_dict(config):
        del config.visualization.layout
    with pytest.raises(ConfigurationError, match="layout"):
        validate_visualization(config)


@pytest.mark.parametrize("width", [0, -1])
def test_nonpositive_overlay_width_is_rejected(width: int) -> None:
    config = _visualization_config(device="cpu")
    config.visualization.overlay.output_width = width
    with pytest.raises(ValueError, match="output_width"):
        build_runtime_config(config)
    with pytest.raises(ConfigurationError, match="output_width"):
        validate_visualization(config)


def test_negative_start_frame_is_rejected_before_loading_data() -> None:
    config = _visualization_config(device="cpu")
    config.visualization.start_frame = -1
    with pytest.raises(ValueError, match="start_frame"):
        build_runtime_config(config)
    with pytest.raises(ConfigurationError, match="start_frame"):
        validate_visualization(config)


@pytest.mark.parametrize("stride", [1, 2, 5])
def test_source_speed_fps_is_divided_by_render_stride(stride: int) -> None:
    source_fps = 2997003 / 50000
    fps = overlay_playback_fps(None, source_fps, stride)
    assert fps == pytest.approx(source_fps / stride)
    assert (1010 / stride) / fps == pytest.approx(1010 / source_fps)
    assert overlay_playback_fps(15, source_fps, stride) == 15


@pytest.mark.parametrize("field,value", [
    ("frame_stride", 0), ("frame_stride", -1), ("styles", []),
    ("styles", ["unknown"]), ("styles", ["ring", "ring"]),
])
def test_invalid_overlay_settings_are_rejected(field: str, value: object) -> None:
    config = _visualization_config(device="cpu")
    config.visualization.overlay[field] = value
    with pytest.raises(ValueError, match=field):
        build_runtime_config(config)
    with pytest.raises(ConfigurationError, match=field):
        validate_visualization(config)


def test_source_speed_requires_overlay_layout() -> None:
    config = _visualization_config(device="cpu")
    config.visualization.fps = None
    with pytest.raises(ValueError, match="requires overlay"):
        build_runtime_config(config)
    with pytest.raises(ConfigurationError, match="requires overlay"):
        validate_visualization(config)


@pytest.mark.parametrize("layout", ["grid_2x2", "overlay"])
def test_real_cpu_prediction_saves_gif(
    tmp_path: Path,
    make_tiny_checkpoint: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """Exercise real inference and exports, including unchanged GIF-only grids."""
    clip_id = "tracknet/game1/Clip1"
    write_store_clip(
        tmp_path / "data/ball_detection/test", clip_id,
        [frame(index) for index in range(5)], size=(640, 360),
    )
    make_tiny_checkpoint(tmp_path / "ckpt/ball_detection/tiny.ckpt")
    config = _visualization_config(device="cpu")
    config.paths.project_root = str(tmp_path)
    config.data.data_dir = "ball_detection/test"
    config.data.image_size = [64, 128]
    config.data.augmentation.normalize_imagenet.enabled = False
    config.model.num_frames = 2
    config.visualization.checkpoint = {"role": "checkpoint", "path": "ball_detection/tiny.ckpt"}
    config.visualization.layout = layout
    config.visualization.overlay.output_width = 320
    config.visualization.save = "ball_detection/presentation/ball_2d/test/ball_2d.gif"
    config.visualization.max_frames = 3
    config.visualization.start_frame = 2
    if layout == "overlay":
        def unexpected_mdd(**_kwargs: object) -> None:
            pytest.fail("overlay must not build the unused MDD panel")
        monkeypatch.setattr(orchestrator, "build_mdd_frames", unexpected_mdd)
    validate_visualization(config)
    runtime = build_runtime_config(config)

    assert orchestrator.run_visualization(runtime) == 0

    with Image.open(runtime.save) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 3
        assert gif.size == ((320, 180) if layout == "overlay" else (268, 232))
        durations = []
        for index in range(gif.n_frames):
            gif.seek(index)
            durations.append(gif.info["duration"])
        assert durations == [80, 90, 80]  # Main's shared cumulative GIF timing.
    manifest_path = runtime.save.parent / "manifest.json"
    if layout == "overlay":
        with av.open(str(runtime.save.with_suffix(".mp4"))) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.name == "h264"
            assert float(stream.average_rate) == 12
            assert len(list(container.decode(stream))) == 3
            assert (stream.width, stream.height) == (320, 180)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["clip_id"] == clip_id
        assert manifest["frame_names"] == [f"{clip_id}:{index}" for index in range(2, 5)]
        assert manifest["frame_range"] == {"start": 2, "stop": 5}
        assert manifest["config"] == OmegaConf.to_container(config, resolve=True)
        assert manifest["elements"] == {"video": "measured", "predictions": "measured"}
        assert len(manifest["git_commit"]) == 40
        assert manifest["overlay_style"] == "ring"
        assert manifest["outputs"]["gif"]["path"] == str(runtime.save)
        assert manifest["outputs"]["mp4"]["path"] == str(runtime.save.with_suffix(".mp4"))
        for entry in [*manifest["inputs"], manifest["checkpoint"], *manifest["outputs"].values()]:
            assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
    else:
        assert not list(runtime.save.parent.glob("*.mp4"))
        assert not manifest_path.exists()


def test_both_overlay_styles_share_full_inference_before_stride(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path],
) -> None:
    clip_id = "meiji/video_000/clip_000/cam2"
    write_store_clip(
        tmp_path / "data/ball_detection/test", clip_id,
        [frame(index) for index in range(6)], size=(128, 72),
    )
    make_tiny_checkpoint(tmp_path / "ckpt/ball_detection/tiny.ckpt")
    config = _visualization_config(device="cpu")
    config.paths.project_root = str(tmp_path)
    config.data.data_dir = "ball_detection/test"
    config.data.image_size = [64, 128]
    config.data.augmentation.normalize_imagenet.enabled = False
    config.model.num_frames = 2
    config.visualization.checkpoint = {"role": "checkpoint", "path": "ball_detection/tiny.ckpt"}
    config.visualization.clip_id = clip_id
    config.visualization.layout = "overlay"
    config.visualization.overlay.output_width = 320
    config.visualization.overlay.frame_stride = 2
    config.visualization.overlay.styles = ["ring", "trail"]
    config.visualization.fps = None
    config.visualization.save = "ball_detection/presentation/ball_2d/test/ball_2d.gif"
    validate_visualization(config)
    runtime = build_runtime_config(config)
    with patch.object(orchestrator, "predict_clip", wraps=orchestrator.predict_clip) as predict:
        assert orchestrator.run_visualization(runtime) == 0
        predict.assert_called_once()
        assert predict.call_args.kwargs["clip"].model_images.shape[0] == 6
    for style in ("ring", "trail"):
        path = runtime.save.parent / style / runtime.save.name
        with Image.open(path) as gif:
            assert gif.n_frames == 3
            assert gif.size == (320, 180)
            duration = 0
            for index in range(gif.n_frames):
                gif.seek(index)
                duration += gif.info["duration"]
            assert duration == 200  # 6 source frames / 30fps, not 3 / 30fps.
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert manifest["overlay_style"] == style
        assert manifest["frame_range"] == {"start": 0, "stop": 6}
        assert manifest["rendered_frame_names"] == [f"{clip_id}:{i}" for i in (0, 2, 4)]
        assert manifest["playback"] == {
            "source_fps": 30, "frame_stride": 2, "gif_fps": 15,
            "inference_frames": 6, "rendered_frames": 3,
        }
        with av.open(str(path.with_suffix(".mp4"))) as container:
            stream = container.streams.video[0]
            assert float(stream.average_rate) == manifest["playback"]["gif_fps"]
            assert stream.codec_context.name == "h264"
            video_frames = list(container.decode(stream))
            assert len(video_frames) == manifest["playback"]["rendered_frames"]
            assert len(video_frames) / float(stream.average_rate) == pytest.approx(duration / 1000)
        assert set(manifest["outputs"]) == {"gif", "mp4"}
        for entry in manifest["outputs"].values():
            assert entry["sha256"] == hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()


def test_runtime_config_rejects_unavailable_cuda_before_predictor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    constructions: list[object] = []
    monkeypatch.setattr(
        orchestrator.BallDetectionPredictor,
        "load_from_checkpoint",
        lambda *_args, **_kwargs: constructions.append(object()),
    )
    raw_main = cast(
        Callable[[DictConfig], int],
        inspect.unwrap(visualize_script.main),
    )

    with pytest.raises(DeviceSelectionError, match="CUDA is unavailable"):
        raw_main(_visualization_config(device="cuda"))

    assert constructions == []
