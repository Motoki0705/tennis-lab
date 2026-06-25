"""Unit tests for the shared animated-GIF writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
