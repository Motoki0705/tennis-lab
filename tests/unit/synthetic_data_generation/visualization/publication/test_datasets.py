"""Unit tests for deterministic publication GIF encoding and selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.synthetic_data_generation.visualization.publication.datasets import (
    _render_selected_gif,
    write_deterministic_gif,
)


def _frames() -> tuple[np.ndarray, ...]:
    return tuple(
        np.full((64, 64, 3), color, dtype=np.uint8)
        for color in ((220, 20, 60), (20, 120, 220), (20, 180, 90))
    )


def test_selected_gif_keeps_both_source_endpoints_and_reopens_deterministically(
    tmp_path: Path,
) -> None:
    frames = _frames()
    frame_order = tuple({"frame_index": index} for index in range(len(frames)))
    output = tmp_path / "selection.gif"

    mapping = _render_selected_gif(
        iter(frames),
        frame_order,
        output,
        frame_indices=(0, 2),
        size=(64, 64),
        duration_ms=40,
    )

    assert tuple(item["source_index"] for item in mapping) == (0, 2)
    with Image.open(output) as image:
        assert image.format == "GIF"
        assert image.size == (64, 64)
        assert image.n_frames == 2
        for index, expected in zip((0, 2), frames[::2], strict=True):
            image.seek(index // 2)
            assert image.info["duration"] == 40
            np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected)

    second = tmp_path / "selection-copy.gif"
    _render_selected_gif(
        iter(frames),
        frame_order,
        second,
        frame_indices=(0, 2),
        size=(64, 64),
        duration_ms=40,
    )
    assert output.read_bytes() == second.read_bytes()


def test_selected_gif_requires_endpoint_inclusive_indices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both source timeline endpoints"):
        _render_selected_gif(
            iter(_frames()),
            tuple({"frame_index": index} for index in range(3)),
            tmp_path / "invalid.gif",
            frame_indices=(1, 2),
            size=(64, 64),
            duration_ms=40,
        )


def test_write_deterministic_gif_rejects_mixed_frame_shapes(tmp_path: Path) -> None:
    frames = _frames()
    frames_with_bad_shape = (*frames, np.zeros((32, 64, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="share one uint8 HxWx3 shape"):
        write_deterministic_gif(
            frames_with_bad_shape,
            tmp_path / "invalid.gif",
            duration_ms=40,
        )
