"""Presentation-order video I/O shared by preparation and local reference tools."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.base.annotation.media import Timeline as Timeline
from src.tasks.base.annotation.media import decode_range as decode_range
from src.tasks.base.annotation.media import encode_video as encode_video
from src.tasks.base.annotation.media import probe_video as probe_video

from .contracts import ClipManifest, sha256_file


def check_clip(video: Path, manifest: ClipManifest) -> Timeline:
    if video.name != manifest.filename or sha256_file(video) != manifest.sha256:
        raise ValueError("clip filename/hash does not match manifest")
    if video.stat().st_size != manifest.bytes:
        raise ValueError("clip byte count does not match manifest")
    timeline = probe_video(video)
    if (timeline.width, timeline.height) != (manifest.width, manifest.height):
        raise ValueError("clip dimensions do not match manifest")
    expected = [
        (
            f.clip_pts * Fraction(manifest.time_base),
            f.duration_pts * Fraction(manifest.time_base),
        )
        for f in manifest.frames
    ]
    actual = [
        (p * timeline.time_base, d * timeline.time_base)
        for p, d in zip(timeline.pts, timeline.durations, strict=True)
    ]
    if actual != expected:
        raise ValueError("clip frame count/PTS/duration does not match manifest")
    return timeline


def extract_frames(
    video: Path,
    manifest: ClipManifest,
    output: Path,
    start: int,
    stop: int,
    crop: tuple[int, int, int, int] | None = None,
) -> list[Path]:
    """Save lossless images; labels occupy added margin, never the source pixels."""
    timeline = check_clip(video, manifest)
    if crop is not None:
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= manifest.width and 0 <= y1 < y2 <= manifest.height):
            raise ValueError("crop must be a positive rectangle inside the image")
    output.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    for index, frame in enumerate(
        decode_range(video, timeline, start, stop), start=start
    ):
        pixels: NDArray[np.uint8] = frame.to_ndarray(format="bgr24").astype(
            np.uint8, copy=False
        )
        if crop is not None:
            pixels = pixels[crop[1] : crop[3], crop[0] : crop[2]]
        mapped = manifest.frames[index]
        seconds = (mapped.source_pts - manifest.source_start_pts) * Fraction(
            manifest.time_base
        )
        text = (
            f"frame={index} source={mapped.source_frame_index} t={float(seconds):.6f}s"
        )
        offset = (
            f"crop origin=({crop[0]},{crop[1]}) size=({pixels.shape[1]},{pixels.shape[0]}); return ORIGINAL pixel xy"
            if crop
            else f"content=({pixels.shape[1]},{pixels.shape[0]}) at (0,0); padding is not image content"
        )
        width = max(
            pixels.shape[1],
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] + 8,
            cv2.getTextSize(offset, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] + 8,
        )
        # Preserve the native crop pixels at the upper left; widen padding,
        # not the content, so even a tiny ball crop keeps its complete labels.
        annotated = cv2.copyMakeBorder(
            pixels,
            0,
            54,
            0,
            width - pixels.shape[1],
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        cv2.putText(
            annotated,
            text,
            (4, pixels.shape[0] + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            offset,
            (4, pixels.shape[0] + 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        destination = output / f"frame_{index:07d}{'_crop' if crop else ''}.png"
        if not cv2.imwrite(str(destination), annotated):
            raise OSError(f"could not write {destination}")
        results.append(destination)
    return results
