"""Smoke tests for qualitative artifact saving (real PNG/GIF rendering).

These exercise the real rendering/encoding path (Pillow GIF, matplotlib
animation, TensorBoard summary emission) on tiny dummy frames. CPU-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tasks.base.training.qualitative_saving import (
    save_qualitative_animation,
    save_qualitative_clip,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _frame(fill: int = 50) -> np.ndarray:
    return np.full((6, 8, 3), fill, dtype=np.uint8)


class _RecordingWriter:
    """Minimal TensorBoard SummaryWriter stand-in recording add_* calls."""

    def __init__(self) -> None:
        self.images: list[tuple[str, tuple[int, ...]]] = []
        self.video_summaries = 0

    def add_image(self, tag: str, img, step: int) -> None:
        self.images.append((tag, tuple(np.asarray(img).shape)))

    def _get_file_writer(self):
        writer = self

        class _FW:
            def add_summary(self, summary, step):
                writer.video_summaries += 1

        return _FW()


def test_single_frame_saves_png_and_logs_image(tmp_path: Path) -> None:
    writer = _RecordingWriter()
    out = save_qualitative_clip(
        frames_rgb=[_frame()],
        artifact_dir=tmp_path,
        name="single",
        tb_writer=writer,
        tag="qual/single",
        global_step=3,
    )
    assert out.suffix == ".png"
    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (8, 6)
    # add_image receives (C, H, W)
    assert writer.images and writer.images[0][1] == (3, 6, 8)


def test_multi_frame_saves_gif_and_logs_video(tmp_path: Path) -> None:
    writer = _RecordingWriter()
    out = save_qualitative_clip(
        frames_rgb=[_frame(10), _frame(200), _frame(120)],
        artifact_dir=tmp_path,
        name="clip",
        tb_writer=writer,
        tag="qual/clip",
        global_step=1,
        fps=8.0,
    )
    assert out.suffix == ".gif"
    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) == 3
    assert writer.video_summaries == 1


def test_clip_works_without_writer(tmp_path: Path) -> None:
    out = save_qualitative_clip(
        frames_rgb=[_frame(), _frame(80)],
        artifact_dir=tmp_path,
        name="nowriter",
        tb_writer=None,
        tag="qual/x",
        global_step=0,
    )
    assert out.exists()


def test_float_frames_clipped_to_uint8(tmp_path: Path) -> None:
    # out-of-range floats must be clipped without raising
    frame: np.ndarray = np.full((4, 4, 3), 300.0, dtype=np.float32)
    out = save_qualitative_clip(
        frames_rgb=[frame],
        artifact_dir=tmp_path,
        name="floaty",
        tb_writer=None,
        tag="t",
        global_step=0,
    )
    assert out.exists()


def test_save_animation_smoke(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])

    def _update(frame: int):
        line.set_data(range(frame + 1), range(frame + 1))
        return (line,)

    anim = FuncAnimation(fig, _update, frames=3, blit=True)
    writer = _RecordingWriter()
    out = save_qualitative_animation(
        animation=anim,
        artifact_dir=tmp_path,
        name="anim",
        tb_writer=writer,
        tag="qual/anim",
        global_step=0,
        fps=5.0,
    )
    assert out.suffix == ".gif"
    assert out.exists()
    assert writer.video_summaries == 1
