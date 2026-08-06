"""Regression tests for MDD panel construction in the visualization API."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import (
    BallInputLayout,
    BallModelInputSpec,
)
from src.tasks.ball_detection.visualization.api.predict import build_mdd_frames
from src.tasks.ball_detection.visualization.io.clip import ClipSequence
from src.tasks.base.model_io import bind_model_io


class _IdentityBallModel(torch.nn.Identity):
    in_channels: int
    num_classes: int

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.num_classes = 1


def _build(
    layout: str, *, num_frames: int = 5, h: int = 6, w: int = 8
) -> list[NDArray[np.uint8]]:
    model = _IdentityBallModel()
    adapter = BallModelIOAdapter(
        BallModelInputSpec(
            model_name="test_model",
            input_mode="rgb",
            input_layout=cast(BallInputLayout, layout),
            in_channels=3,
            num_classes=1,
            configured_frames=num_frames,
            image_size_hw=None,
            minimum_spatial_size=None,
            mdd_gain=1.0,
            mdd_offset=0.0,
        ),
        expected_model_type=_IdentityBallModel,
        minimum_frames=1,
    )
    predictor = BallDetectionPredictor(
        bind_model_io(model, adapter),
        torch.device("cpu"),
        subpixel_refine=False,
    )
    clip = ClipSequence(
        frame_names=tuple(f"frame_{index}.jpg" for index in range(num_frames)),
        render_frames_rgb=tuple(
            torch.zeros(h, w, 3, dtype=torch.uint8) for _ in range(num_frames)
        ),
        model_images=torch.rand(num_frames, 3, h, w),
        gt_coords_px=torch.zeros(num_frames, 2),
        gt_visibility=torch.zeros(num_frames, dtype=torch.bool),
    )
    return cast(
        list[NDArray[np.uint8]],
        build_mdd_frames(predictor=predictor, clip=clip),
    )


def test_btchw_model_yields_one_mdd_frame_per_input_frame() -> None:
    """btchw models must not collapse the MDD panel to the channel count."""
    frames = _build("btchw", num_frames=5, h=6, w=8)
    assert len(frames) == 5
    assert frames[0].shape == (6, 8, 3)


def test_mdd_frame_count_is_layout_independent() -> None:
    """bcthw and btchw layouts must produce the same number of MDD frames."""
    assert len(_build("bcthw", num_frames=5)) == len(_build("btchw", num_frames=5))
