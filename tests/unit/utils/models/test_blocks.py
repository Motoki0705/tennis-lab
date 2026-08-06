from __future__ import annotations

import torch

from src.utils.models.blocks import Conv2dWiseWiseBlock, DepthwiseSeparableConv2d


def test_depthwise_separable_conv2d_preserves_spatial_shape() -> None:
    block = DepthwiseSeparableConv2d(3, 8)
    out = block(torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 8, 16, 16)


def test_conv2d_wisewise_block_preserves_spatial_shape() -> None:
    block = Conv2dWiseWiseBlock(4, 6)
    out = block(torch.randn(2, 4, 12, 12))
    assert out.shape == (2, 6, 12, 12)


def test_task_modules_use_the_shared_classes_directly() -> None:
    """Task modules import the canonical utils blocks without local shims."""
    from src.tasks.ball_detection.models.spatiotemporal_unet import (
        Conv2dWiseWiseBlock as BallWiseWise,
    )
    from src.tasks.court_detection.models.decoder import (
        Conv2dWiseWiseBlock as DecoderWiseWise,
    )
    from src.tasks.court_detection.models.encoders import (
        Conv2dWiseWiseBlock as EncoderWiseWise,
    )

    assert DecoderWiseWise is EncoderWiseWise is Conv2dWiseWiseBlock
    assert BallWiseWise is Conv2dWiseWiseBlock
