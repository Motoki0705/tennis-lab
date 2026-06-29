"""Reusable convolution blocks for court detection models.

The implementations now live in :mod:`src.utils.models.blocks` (they are
byte-identical to the ball-detection copies); re-exported here to preserve the
existing ``src.tasks.court_detection.models.blocks`` import path.
"""

from __future__ import annotations

from src.utils.models.blocks import Conv2dWiseWiseBlock, DepthwiseSeparableConv2d

__all__ = ["Conv2dWiseWiseBlock", "DepthwiseSeparableConv2d"]
