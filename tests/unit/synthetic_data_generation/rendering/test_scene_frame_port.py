"""Unit tests for the reconstructed static-scene frame contract."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic_data_generation.rendering.scene_frame_port import SceneFrame


def test_scene_frame_rejects_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="positive or infinity"):
        SceneFrame(
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth=np.zeros((2, 3), dtype=np.float32),
            alpha=np.ones((2, 3), dtype=np.float32),
            scene_fingerprint="a" * 64,
            camera_id="camera",
            backend_id="backend",
            backend_version="version",
        )


def test_scene_frame_accepts_infinite_empty_depth_and_freezes_arrays() -> None:
    frame = SceneFrame(
        rgb=np.zeros((2, 3, 3), dtype=np.uint8),
        depth=np.full((2, 3), np.inf, dtype=np.float32),
        alpha=np.zeros((2, 3), dtype=np.float32),
        scene_fingerprint="a" * 64,
        camera_id="camera",
        backend_id="backend",
        backend_version="version",
    )

    assert not frame.rgb.flags.writeable
    assert not frame.depth.flags.writeable
    assert not frame.alpha.flags.writeable
