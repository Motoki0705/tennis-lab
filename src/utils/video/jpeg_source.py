"""Reproduce pinned JPEG inputs from an immutable video without storing RGB."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import cv2

from src.utils.data.verification import FileVersion, file_version
from src.utils.video.encoding import encode_jpeg


@lru_cache(maxsize=1)
def jpeg_video_runtime() -> str:
    """Pin the decoder, color conversion and JPEG encoder implementations."""
    value = {
        "algorithm": "pyav-seek-bgr24-opencv-jpeg.v1",
        "av": av.__version__,
        "ffmpeg": av.library_versions,
        "opencv": cv2.__version__,
        "opencv_build": cv2.getBuildInformation(),
        "decoder_threads": 2,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_video_jpegs(
    path: Path,
    indices: Sequence[int],
    *,
    pts: Sequence[int],
    time_base: Fraction,
    width: int,
    height: int,
    quality: int,
    runtime: str,
    frame_hashes: Sequence[str],
    version: FileVersion,
) -> tuple[bytes, ...]:
    """Read at most 32 ordered frames and verify their exact publication bytes.

    Callers verify the immutable video's content hash before passing its stat
    version. No resizing, nearest-frame lookup, decoder fallback or disk cache.
    Only selected JPEG payloads and decoder buffers are retained in memory.
    """
    if runtime != jpeg_video_runtime():
        raise ValueError("Video JPEG runtime differs from the publication runtime")
    if (
        not 1 <= len(indices) <= 32
        or any(a >= b for a, b in zip(indices, indices[1:], strict=False))
        or indices[0] < 0
        or indices[-1] >= len(pts)
    ):
        raise ValueError("Video JPEG selection requires 1..32 ordered frame indices")
    if (
        len(frame_hashes) != len(pts)
        or time_base <= 0
        or width <= 0
        or height <= 0
        or any(a >= b for a, b in zip(pts, pts[1:], strict=False))
    ):
        raise ValueError("Invalid pinned video timeline/geometry")
    if file_version(path) != version:
        raise ValueError("Video changed before JPEG read")
    wanted = {pts[index]: index for index in indices}
    all_points = set(pts)
    result: dict[int, bytes] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if stream.time_base is None:
            raise ValueError("Video has no stream time base")
        offset = Fraction(pts[indices[0]]) * time_base / Fraction(stream.time_base)
        if offset.denominator != 1:
            raise ValueError("Video seek timestamp is not exact")
        container.seek(int(offset), backward=True, any_frame=False, stream=stream)
        previous: int | None = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Video frame has no presentation timestamp")
            point = Fraction(frame.pts) * Fraction(frame.time_base) / time_base
            if point.denominator != 1 or int(point) not in all_points:
                raise ValueError("Decoded video PTS differs from the pinned timeline")
            timestamp = int(point)
            if previous is not None and timestamp <= previous:
                raise ValueError("Decoded video timestamps are not strictly increasing")
            previous = timestamp
            if (frame.width, frame.height) != (width, height):
                raise ValueError("Decoded video dimensions differ from publication")
            if timestamp in wanted:
                payload = encode_jpeg(frame.to_ndarray(format="bgr24"), quality=quality)
                if (
                    hashlib.sha256(payload).hexdigest()
                    != frame_hashes[wanted[timestamp]]
                ):
                    raise ValueError("Regenerated JPEG differs from publication bytes")
                result[timestamp] = payload
            if timestamp >= pts[indices[-1]]:
                break
    if file_version(path) != version:
        raise ValueError("Video changed during JPEG read")
    if set(result) != set(wanted):
        raise ValueError("Video did not provide every requested frame")
    return tuple(result[pts[index]] for index in indices)
