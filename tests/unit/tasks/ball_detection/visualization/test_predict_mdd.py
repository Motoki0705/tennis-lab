"""Regression tests for MDD panel construction in the visualization API."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.tasks.ball_detection.visualization.api.predict import build_mdd_frames


def _build(layout: str, *, num_frames: int = 5, h: int = 6, w: int = 8) -> list:
    predictor = SimpleNamespace(
        model_config={
            "input_mode": "rgb",
            "input_layout": layout,
            "in_channels": 3,
            "num_frames": num_frames,
        }
    )
    clip = SimpleNamespace(model_images=torch.rand(num_frames, 3, h, w))
    return build_mdd_frames(predictor=predictor, clip=clip)


def test_btchw_model_yields_one_mdd_frame_per_input_frame() -> None:
    """btchw models must not collapse the MDD panel to the channel count."""
    frames = _build("btchw", num_frames=5, h=6, w=8)
    assert len(frames) == 5
    assert frames[0].shape == (6, 8, 3)


def test_mdd_frame_count_is_layout_independent() -> None:
    """bcthw and btchw layouts must produce the same number of MDD frames."""
    assert len(_build("bcthw", num_frames=5)) == len(_build("btchw", num_frames=5))
