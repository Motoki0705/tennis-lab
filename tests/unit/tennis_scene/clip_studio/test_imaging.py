"""Tests for src/tennis_scene/clip_studio/imaging.py."""

import numpy as np
import pytest

from src.tennis_scene.clip_studio.imaging import compute_letterbox, letterbox_frame


class TestComputeLetterbox:
    def test_exact_fit(self) -> None:
        spec = compute_letterbox(64, 48, 64, 48)
        assert spec.scale == 1.0
        assert (spec.pad_x, spec.pad_y) == (0, 0)
        assert (spec.scaled_width, spec.scaled_height) == (64, 48)

    def test_width_limited(self) -> None:
        # 64x48 into 96x64: height limits (64/48 < 96/64)
        spec = compute_letterbox(64, 48, 96, 64)
        assert spec.scale == pytest.approx(64 / 48)
        assert (spec.scaled_width, spec.scaled_height) == (85, 64)
        assert (spec.pad_x, spec.pad_y) == (5, 0)

    def test_height_limited(self) -> None:
        spec = compute_letterbox(100, 50, 50, 50)
        assert spec.scale == pytest.approx(0.5)
        assert (spec.scaled_width, spec.scaled_height) == (50, 25)
        assert (spec.pad_x, spec.pad_y) == (0, 12)

    def test_invalid_sizes_raise(self) -> None:
        with pytest.raises(ValueError, match="source size"):
            compute_letterbox(0, 48, 64, 48)
        with pytest.raises(ValueError, match="target size"):
            compute_letterbox(64, 48, 64, 0)


class TestLetterboxFrame:
    def test_content_centered_and_padded(self) -> None:
        frame: np.ndarray = np.full((50, 100, 3), 200, dtype=np.uint8)
        fitted, spec = letterbox_frame(frame, 50, 50)
        assert fitted.shape == (50, 50, 3)
        assert (spec.pad_x, spec.pad_y) == (0, 12)
        # padding rows are fill_value, content rows keep the source intensity
        assert int(fitted[:12].max()) == 0
        assert int(fitted[12 : 12 + 25].min()) == 200
        assert int(fitted[12 + 25 :].max()) == 0

    def test_no_resize_when_exact(self) -> None:
        frame: np.ndarray = np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
        fitted, spec = letterbox_frame(frame, 64, 48)
        assert np.array_equal(fitted, frame)
        assert spec.scale == 1.0

    def test_invalid_frame_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            letterbox_frame(np.zeros((10, 10), dtype=np.uint8), 20, 20)
