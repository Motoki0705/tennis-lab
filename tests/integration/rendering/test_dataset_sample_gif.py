"""Smoke test for atomic dataset-sample GIF encoding and validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from PIL import Image

from src.tasks.base.generate_dataset.dataset_samples import save_animation_gif


def test_dataset_sample_gif_writer_encodes_every_frame_and_closes_figure(
    tmp_path: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(2.0, 2.0))
    (line,) = axes.plot([], [])

    def update(frame: int) -> tuple[Artist, ...]:
        line.set_data([0, frame], [frame, 0])
        axes.set_xlim(0, 3)
        axes.set_ylim(0, 3)
        return (line,)

    animation = FuncAnimation(figure, update, frames=4, blit=False)
    output = tmp_path / "sample.gif"

    save_animation_gif(animation, path=output, fps=8, expected_frames=4)

    with Image.open(output) as image:
        assert getattr(image, "n_frames", 1) == 4
        assert image.size == (200, 200)
    assert not plt.fignum_exists(figure.number)
