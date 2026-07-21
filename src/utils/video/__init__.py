"""Shared video streaming utilities."""

from src.utils.video.animation import GifEncodingOptions, save_rgb_animation
from src.utils.video.audio import audio_envelope, read_audio_mono
from src.utils.video.batching import iter_temporal_batches
from src.utils.video.encoding import encode_jpeg, iter_selected_video_jpegs
from src.utils.video.prefetch import PrefetchIterator
from src.utils.video.reader import (
    OpenCVVideoFrameReader,
    RandomAccessVideoReader,
    probe_video_info,
    read_video_frame,
    read_video_rgb,
)
from src.utils.video.sampling import (
    parse_time_seconds,
    sample_frame_indices_by_time_ranges,
    sample_step_seconds,
    sample_uniform_frame_indices,
)
from src.utils.video.transforms import BgrToTensorTransform, normalize_tensor_imagenet
from src.utils.video.types import FramePacket, TemporalBatch, TemporalWindow, VideoInfo
from src.utils.video.windows import (
    build_window_starts,
    chunked,
    iter_temporal_windows,
)
from src.utils.video.writer import VideoWriter, save_video_rgb
from src.utils.video.youtube import (
    download_youtube_video,
    find_downloaded_video,
    h264_encoder_args,
    transcode_h264_video,
)

__all__ = [
    "audio_envelope",
    "build_window_starts",
    "chunked",
    "BgrToTensorTransform",
    "encode_jpeg",
    "FramePacket",
    "GifEncodingOptions",
    "OpenCVVideoFrameReader",
    "RandomAccessVideoReader",
    "PrefetchIterator",
    "TemporalBatch",
    "TemporalWindow",
    "VideoInfo",
    "VideoWriter",
    "iter_temporal_batches",
    "iter_temporal_windows",
    "iter_selected_video_jpegs",
    "download_youtube_video",
    "find_downloaded_video",
    "h264_encoder_args",
    "normalize_tensor_imagenet",
    "parse_time_seconds",
    "probe_video_info",
    "read_audio_mono",
    "read_video_frame",
    "read_video_rgb",
    "sample_frame_indices_by_time_ranges",
    "save_rgb_animation",
    "save_video_rgb",
    "sample_step_seconds",
    "sample_uniform_frame_indices",
    "transcode_h264_video",
]
