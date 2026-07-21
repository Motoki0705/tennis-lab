"""Tests for shared GIF/MP4 animation encoding."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils.video.animation import GifEncodingOptions, save_rgb_animation
from src.utils.video.reader import probe_video_info, read_video_rgb

pytestmark = pytest.mark.unit


def _frames(
    *,
    count: int = 3,
    height: int = 32,
    width: int = 48,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = 30 + index * 60
        frame[..., 1] = 120
        frames.append(frame)
    return frames


def test_empty_frames_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one frame"):
        save_rgb_animation([], tmp_path / "empty.gif", fps=10)


def test_invalid_fps_and_loop_raise(tmp_path: Path) -> None:
    frames = _frames(count=1)
    with pytest.raises(ValueError, match="fps must be positive"):
        save_rgb_animation(frames, tmp_path / "bad.gif", fps=0)
    with pytest.raises(ValueError, match="loop must be non-negative"):
        save_rgb_animation(frames, tmp_path / "bad.gif", fps=10, loop=-1)


def test_rejects_unsupported_suffix_and_inconsistent_frames(tmp_path: Path) -> None:
    frames = _frames(count=1)
    with pytest.raises(ValueError, match="expected .gif or .mp4"):
        save_rgb_animation(frames, tmp_path / "bad.webm", fps=10)

    inconsistent = [frames[0], np.zeros((16, 24, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="frame size changed"):
        save_rgb_animation(inconsistent, tmp_path / "bad.gif", fps=10)


def test_writes_multi_frame_gif_with_shared_palette(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "animation.gif"
    save_rgb_animation(
        _frames(),
        output,
        fps=12,
        loop=2,
        gif_options=GifEncodingOptions(
            colors=32,
            palette_sample_frames=3,
            palette_max_size=16,
        ),
    )

    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "GIF"
        assert image.size == (48, 32)
        assert image.n_frames == 3
        assert image.info["loop"] == 2
        assert image.info["duration"] == pytest.approx(1000 / 12, abs=10)


def test_invalid_gif_options_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="colors"):
        save_rgb_animation(
            _frames(count=1),
            tmp_path / "bad.gif",
            fps=10,
            gif_options=GifEncodingOptions(colors=1),
        )


def test_writes_mp4_and_pads_odd_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "animation.mp4"
    save_rgb_animation(_frames(height=31, width=47), output, fps=10, video_crf=10)

    info = probe_video_info(output)
    assert (info.width, info.height) == (48, 32)
    assert info.frame_count == 3
    assert info.fps == pytest.approx(10.0, abs=0.1)

    decoded = read_video_rgb(output)
    assert decoded.shape == (3, 32, 48, 3)
