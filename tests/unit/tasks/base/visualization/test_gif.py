"""Unit tests for the shared animated-GIF writer."""

from __future__ import annotations

from itertools import accumulate
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence

from src.tasks.base.visualization.gif import save_gif

pytestmark = pytest.mark.unit


def _frame(h: int = 4, w: int = 6, fill: int = 30) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_empty_frames_raises() -> None:
    with pytest.raises(ValueError, match="At least one frame"):
        save_gif(frames_rgb=[], path=Path("/tmp/x.gif"), fps=10)


def test_non_positive_fps_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        save_gif(frames_rgb=[_frame()], path=tmp_path / "x.gif", fps=0)
    with pytest.raises(ValueError, match="fps must be positive"):
        save_gif(frames_rgb=[_frame()], path=tmp_path / "x.gif", fps=-5)


def test_non_gif_suffix_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only .gif"):
        save_gif(frames_rgb=[_frame()], path=tmp_path / "x.png", fps=10)


def test_writes_multi_frame_gif(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "anim.gif"
    frames = [_frame(fill=10), _frame(fill=200), _frame(fill=120)]
    save_gif(frames_rgb=frames, path=out, fps=12)
    assert out.exists()
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert im.size == (6, 4)  # (width, height)
        n = getattr(im, "n_frames", 1)
        assert n == 3


def test_creates_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c.gif"
    save_gif(frames_rgb=[_frame()], path=out, fps=10)
    assert out.parent.is_dir()


def _decoded_durations_ms(path: Path) -> list[int]:
    with Image.open(path) as gif:
        durations: list[int] = [
            frame.info["duration"] for frame in ImageSequence.Iterator(gif)
        ]
    return durations


@pytest.mark.parametrize("fps,frame_count", [(15, 900), (15, 7), (10, 600), (10, 7)])
def test_decoded_gif_preserves_total_duration_and_frame_times(
    tmp_path: Path,
    fps: float,
    frame_count: int,
) -> None:
    out = tmp_path / "timed.gif"
    frames = [
        _frame(fill=30 if index % 2 == 0 else 200) for index in range(frame_count)
    ]
    save_gif(frames_rgb=frames, path=out, fps=fps)

    durations = _decoded_durations_ms(out)
    assert len(durations) == frame_count  # Different adjacent images cannot merge.
    assert abs(sum(durations) - frame_count / fps * 1000) < 10
    for end_frame, elapsed_ms in enumerate(accumulate(durations), start=1):
        assert abs(elapsed_ms - end_frame / fps * 1000) < 10


@pytest.mark.parametrize("fps", [15, 10])
@pytest.mark.parametrize("run_lengths", [(4, 4, 4), (900,)])
def test_merged_identical_frames_preserve_cumulative_duration(
    tmp_path: Path,
    fps: float,
    run_lengths: tuple[int, ...],
) -> None:
    out = tmp_path / "merged.gif"
    frames = [
        _frame(fill=30 + run_index * 50)
        for run_index, count in enumerate(run_lengths)
        for _ in range(count)
    ]
    save_gif(frames_rgb=frames, path=out, fps=fps)

    durations = _decoded_durations_ms(out)
    assert len(durations) == len(run_lengths)  # Pillow actually merged the runs.
    assert abs(sum(durations) - len(frames) / fps * 1000) < 10
    for end_frame, elapsed_ms in zip(
        accumulate(run_lengths),
        accumulate(durations),
        strict=True,
    ):
        assert abs(elapsed_ms - end_frame / fps * 1000) < 10
