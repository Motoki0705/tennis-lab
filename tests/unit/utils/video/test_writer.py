"""Tests for src/utils/video/writer.py (and read_video_rgb round-trip)."""

from pathlib import Path

import numpy as np
import pytest

from src.utils.video.reader import probe_video_info, read_video_rgb
from src.utils.video.writer import VideoWriter, save_video_rgb


def make_frames(n: int = 6, height: int = 32, width: int = 48) -> np.ndarray:
    """Solid-color frames with per-frame distinct intensity."""
    frames: np.ndarray = np.zeros((n, height, width, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, :, :, 0] = 30 + i * 30
        frames[i, :, :, 1] = 120
    return frames


class TestVideoWriter:
    def test_write_and_read_back(self, tmp_path: Path) -> None:
        frames = make_frames()
        out = tmp_path / "out.mp4"
        save_video_rgb(frames, out, fps=10.0, crf=10)

        info = probe_video_info(out)
        assert (info.width, info.height) == (48, 32)
        assert info.frame_count == len(frames)
        assert info.fps == pytest.approx(10.0, abs=0.1)

        decoded = read_video_rgb(out)
        assert decoded.shape == frames.shape
        # Lossy codec: solid-color frames should still be close.
        assert np.abs(decoded.astype(int) - frames.astype(int)).mean() < 8

    def test_context_manager_and_streaming(self, tmp_path: Path) -> None:
        out = tmp_path / "stream.mp4"
        frames = make_frames(n=4)
        with VideoWriter(out, fps=5.0) as writer:
            for frame in frames:
                writer.write_frame(frame)
        assert probe_video_info(out).frame_count == 4

    def test_close_idempotent(self, tmp_path: Path) -> None:
        writer = VideoWriter(tmp_path / "x.mp4", fps=5.0)
        writer.write_frame(make_frames(n=1)[0])
        writer.close()
        writer.close()
        with pytest.raises(RuntimeError, match="already closed"):
            writer.write_frame(make_frames(n=1)[0])

    def test_odd_dimensions_rejected(self, tmp_path: Path) -> None:
        writer = VideoWriter(tmp_path / "odd.mp4", fps=5.0)
        frame: np.ndarray = np.zeros((31, 48, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="even frame dimensions"):
            writer.write_frame(frame)

    def test_inconsistent_size_rejected(self, tmp_path: Path) -> None:
        with VideoWriter(tmp_path / "sz.mp4", fps=5.0) as writer:
            writer.write_frame(np.zeros((32, 48, 3), dtype=np.uint8))
            with pytest.raises(ValueError, match="frame size changed"):
                writer.write_frame(np.zeros((32, 64, 3), dtype=np.uint8))

    def test_invalid_frames_rejected(self, tmp_path: Path) -> None:
        writer = VideoWriter(tmp_path / "bad.mp4", fps=5.0)
        with pytest.raises(ValueError, match="shape"):
            writer.write_frame(np.zeros((32, 48), dtype=np.uint8))
        with pytest.raises(ValueError, match="uint8"):
            writer.write_frame(np.zeros((32, 48, 3), dtype=np.float32))

    def test_save_video_rgb_validates_shape(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="N, H, W, 3"):
            save_video_rgb(np.zeros((32, 48, 3), dtype=np.uint8), tmp_path / "v.mp4")


class TestReadVideoRgb:
    def test_scale_and_max_frames(self, tmp_path: Path) -> None:
        frames = make_frames(n=6, height=32, width=48)
        out = tmp_path / "scaled.mp4"
        save_video_rgb(frames, out, fps=10.0)

        decoded = read_video_rgb(out, max_frames=3, scale=0.5)
        assert decoded.shape == (3, 16, 24, 3)

    def test_invalid_scale(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale must be positive"):
            read_video_rgb(tmp_path / "none.mp4", scale=0.0)

    def test_missing_video_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            read_video_rgb(tmp_path / "missing.mp4")
