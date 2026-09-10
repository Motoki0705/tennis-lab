"""Tests for the one-trajectory Court pose evaluation boundary."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_detection.scripts.evaluate_pose_trajectory import _load_rgb


def test_load_rgb_reads_strict_synthetic_court_v3_array(tmp_path) -> None:
    image_path = tmp_path / "rgb.npy"
    rgb: NDArray[np.float32] = np.full((3, 5, 3), 0.5, dtype=np.float32)
    np.save(image_path, rgb)

    image = _load_rgb(image_path, sample={"height": 3, "width": 5})

    assert image.mode == "RGB"
    assert image.size == (5, 3)
    assert np.asarray(image).tolist() == np.full((3, 5, 3), 128, dtype=np.uint8).tolist()


def test_load_rgb_rejects_shape_disagreement(tmp_path) -> None:
    image_path = tmp_path / "rgb.npy"
    np.save(image_path, np.zeros((3, 5, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="matching the manifest"):
        _load_rgb(image_path, sample={"height": 4, "width": 5})
