"""H.264 video writing utilities backed by PyAV.

Complements :mod:`src.utils.video.reader` (OpenCV-based reading) with an
encoder that gives explicit control over fps and CRF quality.

Example:
    >>> with VideoWriter("out.mp4", fps=30.0) as writer:
    ...     for frame in frames_rgb:  # (H, W, 3) uint8
    ...         writer.write_frame(frame)
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VideoWriter:
    """Streaming H.264 (libx264, yuv420p) video writer for RGB uint8 frames.

    The stream is initialized lazily from the first frame's size. Frame
    dimensions must be even (yuv420p chroma subsampling requirement) and
    consistent across frames.

    Args:
        video_path: Output file path (container inferred from suffix, e.g. .mp4).
        fps: Output frame rate.
        crf: x264 constant rate factor; 17 is visually lossless, 23 is the
            x264 default, +6 roughly halves the bitrate.
    """

    def __init__(self, video_path: str | Path, fps: float = 30.0, crf: int = 17) -> None:
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(video_path), mode="w")
        self._fps = Fraction(fps).limit_denominator(1_000_000)
        self._crf = crf
        self._stream: av.VideoStream | None = None
        self._closed = False

    def _init_stream(self, height: int, width: int) -> av.VideoStream:
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"libx264/yuv420p requires even frame dimensions, got {width}x{height}"
            )
        stream = self._container.add_stream("libx264", rate=self._fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(self._crf)}
        return stream

    def write_frame(self, frame_rgb: NDArray[np.uint8]) -> None:
        """Encode one RGB frame of shape (H, W, 3) uint8."""
        if self._closed:
            raise RuntimeError("VideoWriter is already closed")
        frame_rgb = np.ascontiguousarray(frame_rgb)
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"frame must have shape (H, W, 3), got {frame_rgb.shape}")
        if frame_rgb.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {frame_rgb.dtype}")

        height, width = frame_rgb.shape[:2]
        if self._stream is None:
            self._stream = self._init_stream(height, width)
        elif (self._stream.height, self._stream.width) != (height, width):
            raise ValueError(
                f"frame size changed from {self._stream.width}x{self._stream.height} "
                f"to {width}x{height}"
            )

        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)

    def close(self) -> None:
        """Flush the encoder and finalize the container (idempotent)."""
        if self._closed:
            return
        if self._stream is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
        self._container.close()
        self._closed = True

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def save_video_rgb(
    frames_rgb: NDArray[np.uint8],
    video_path: str | Path,
    *,
    fps: float = 30.0,
    crf: int = 17,
) -> None:
    """Write a batch of RGB frames (N, H, W, 3) uint8 to an H.264 video."""
    frames_rgb = np.asarray(frames_rgb)
    if frames_rgb.ndim != 4 or frames_rgb.shape[3] != 3:
        raise ValueError(f"frames must have shape (N, H, W, 3), got {frames_rgb.shape}")
    with VideoWriter(video_path, fps=fps, crf=crf) as writer:
        for frame in frames_rgb:
            writer.write_frame(frame)
