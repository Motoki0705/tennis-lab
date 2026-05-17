"""Shared video streaming utilities."""

from src.utils.video.batching import iter_temporal_batches
from src.utils.video.prefetch import PrefetchIterator
from src.utils.video.reader import OpenCVVideoFrameReader, probe_video_info
from src.utils.video.transforms import BgrToTensorTransform, normalize_tensor_imagenet
from src.utils.video.types import FramePacket, TemporalBatch, TemporalWindow, VideoInfo
from src.utils.video.windows import iter_temporal_windows

__all__ = [
    "BgrToTensorTransform",
    "FramePacket",
    "OpenCVVideoFrameReader",
    "PrefetchIterator",
    "TemporalBatch",
    "TemporalWindow",
    "VideoInfo",
    "iter_temporal_batches",
    "iter_temporal_windows",
    "normalize_tensor_imagenet",
    "probe_video_info",
]
