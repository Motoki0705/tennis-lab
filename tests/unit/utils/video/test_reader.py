"""Tests for src/utils/video/reader.py (RandomAccessVideoReader)."""

from pathlib import Path

import numpy as np
import pytest

from src.utils.video.reader import (
    OpenCVVideoFrameReader,
    RandomAccessVideoReader,
    VideoReadError,
    probe_video_decoder,
    probe_video_info,
    read_video_frame,
    read_video_rgb,
)
from src.utils.video.writer import save_video_rgb


def make_indexed_video(path: Path, n: int = 24) -> None:
    """Write a video whose frame i is solid red intensity ``i * 10``."""
    frames: np.ndarray = np.zeros((n, 32, 48, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, :, :, 0] = i * 10
    save_video_rgb(frames, path, fps=10.0, crf=10)


def decode_index(frame_bgr: np.ndarray) -> int:
    """Recover the frame index from a frame written by make_indexed_video."""
    return int(round(float(frame_bgr[:, :, 2].mean()) / 10))


class TestRandomAccessVideoReader:
    def test_sequential_reads(self, tmp_path: Path) -> None:
        video = tmp_path / "seq.mp4"
        make_indexed_video(video)
        with RandomAccessVideoReader(video) as reader:
            for i in range(5):
                assert decode_index(reader.read(i)) == i

    def test_scattered_reads(self, tmp_path: Path) -> None:
        video = tmp_path / "scatter.mp4"
        make_indexed_video(video)
        with RandomAccessVideoReader(video, seek_grab_threshold=4) as reader:
            # forward grab, backward seek, long forward seek, repeat frame
            for index in [0, 3, 1, 20, 20, 7]:
                assert decode_index(reader.read(index)) == index

    def test_read_same_frame_twice(self, tmp_path: Path) -> None:
        video = tmp_path / "same.mp4"
        make_indexed_video(video)
        with RandomAccessVideoReader(video) as reader:
            assert decode_index(reader.read(5)) == 5
            assert decode_index(reader.read(5)) == 5

    def test_negative_index_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "neg.mp4"
        make_indexed_video(video)
        with (
            RandomAccessVideoReader(video) as reader,
            pytest.raises(ValueError, match="non-negative"),
        ):
            reader.read(-1)

    def test_out_of_range_read_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "oob.mp4"
        make_indexed_video(video, n=6)
        with (
            RandomAccessVideoReader(video) as reader,
            pytest.raises(VideoReadError, match="Failed to read frame"),
        ):
            reader.read(1000)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with (
            RandomAccessVideoReader(tmp_path / "missing.mp4") as reader,
            pytest.raises(VideoReadError, match="Failed to open"),
        ):
            reader.read(0)

    def test_invalid_threshold_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="seek_grab_threshold"):
            RandomAccessVideoReader(tmp_path / "x.mp4", seek_grab_threshold=-1)

    def test_close_idempotent(self, tmp_path: Path) -> None:
        video = tmp_path / "close.mp4"
        make_indexed_video(video, n=4)
        reader = RandomAccessVideoReader(video)
        reader.read(0)
        reader.close()
        reader.close()


@pytest.mark.parametrize("read", [probe_video_info, probe_video_decoder, lambda path: read_video_frame(path, 0),
    lambda path: next(iter(OpenCVVideoFrameReader(path))), read_video_rgb], ids=["probe", "decoder", "frame", "stream", "rgb"])
def test_unreadable_media_has_a_distinct_error_type(tmp_path: Path, read) -> None:
    with pytest.raises(VideoReadError, match="Failed to open"):
        read(tmp_path / "missing.mp4")
