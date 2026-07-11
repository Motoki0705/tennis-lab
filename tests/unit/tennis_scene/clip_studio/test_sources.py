"""Tests for src/tennis_scene/clip_studio/sources.py."""

from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.clip_studio.sources import PreviewSource, PreviewSourcePool
from src.utils.video.writer import save_video_rgb


def make_indexed_video(path: Path, n: int = 20, width: int = 96, height: int = 48) -> None:
    """Frame i is solid red intensity ``i * 12``."""
    frames: np.ndarray = np.zeros((n, height, width, 3), dtype=np.uint8)
    for i in range(n):
        frames[i, :, :, 0] = i * 12
    save_video_rgb(frames, path, fps=10.0, crf=10)


def decode_index(frame_bgr: np.ndarray) -> int:
    return int(round(float(frame_bgr[:, :, 2].mean()) / 12))


class TestPreviewSource:
    def test_downscales_to_tile_width(self, tmp_path: Path) -> None:
        video = tmp_path / "v.mp4"
        make_indexed_video(video)
        with PreviewSource(video, tile_width=48) as source:
            frame = source.get_frame(3)
            assert frame.shape == (24, 48, 3)
            assert decode_index(frame) == 3

    def test_keeps_native_size_when_smaller(self, tmp_path: Path) -> None:
        video = tmp_path / "small.mp4"
        make_indexed_video(video, width=32, height=16)
        with PreviewSource(video, tile_width=640) as source:
            assert source.get_frame(0).shape == (16, 32, 3)

    def test_cache_serves_backward_scrub(self, tmp_path: Path) -> None:
        video = tmp_path / "scrub.mp4"
        make_indexed_video(video)
        with PreviewSource(video, tile_width=48, cache_frames=8) as source:
            forward = [decode_index(source.get_frame(i)) for i in range(6)]
            backward = [decode_index(source.get_frame(i)) for i in reversed(range(6))]
            assert forward == list(range(6))
            assert backward == list(reversed(range(6)))

    def test_cache_eviction_keeps_working(self, tmp_path: Path) -> None:
        video = tmp_path / "evict.mp4"
        make_indexed_video(video)
        with PreviewSource(video, tile_width=48, cache_frames=2) as source:
            for i in [0, 5, 10, 0, 5]:
                assert decode_index(source.get_frame(i)) == i

    def test_out_of_range_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "oob.mp4"
        make_indexed_video(video, n=5)
        with (
            PreviewSource(video) as source,
            pytest.raises(ValueError, match="out of range"),
        ):
            source.get_frame(5)

    def test_invalid_args_raise(self, tmp_path: Path) -> None:
        video = tmp_path / "args.mp4"
        make_indexed_video(video, n=2)
        with pytest.raises(ValueError, match="tile_width"):
            PreviewSource(video, tile_width=0)
        with pytest.raises(ValueError, match="cache_frames"):
            PreviewSource(video, cache_frames=0)


class TestPreviewSourcePool:
    def test_parallel_fetch_with_none(self, tmp_path: Path) -> None:
        video_a = tmp_path / "a.mp4"
        video_b = tmp_path / "b.mp4"
        make_indexed_video(video_a)
        make_indexed_video(video_b)
        with PreviewSourcePool(
            [PreviewSource(video_a, tile_width=48), PreviewSource(video_b, tile_width=48)]
        ) as pool:
            frames = pool.fetch([2, None])
            assert frames[0] is not None
            assert decode_index(frames[0]) == 2
            assert frames[1] is None

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "one.mp4"
        make_indexed_video(video, n=3)
        with (
            PreviewSourcePool([PreviewSource(video)]) as pool,
            pytest.raises(ValueError, match="must match sources"),
        ):
            pool.fetch([0, 1])

    def test_empty_sources_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PreviewSourcePool([])
