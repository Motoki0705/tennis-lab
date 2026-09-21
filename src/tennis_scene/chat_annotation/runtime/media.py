"""Presentation-order video I/O shared by preparation and the portable kit."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import ClipManifest, sha256_file

HDR_TRANSFER_CHARACTERISTICS = {16: "PQ", 18: "HLG"}
COLOR_ATTRIBUTES = ("colorspace", "color_range", "color_primaries", "color_trc")


def _reject_hdr(color_trc: int, path: Path) -> None:
    transfer = HDR_TRANSFER_CHARACTERISTICS.get(color_trc)
    if transfer is not None:
        raise ValueError(
            f"HDR {transfer} input is not supported for 8-bit annotation output: {path}"
        )


@dataclass(frozen=True)
class Timeline:
    width: int
    height: int
    time_base: Fraction
    rate: Fraction
    pts: tuple[int, ...]
    durations: tuple[int, ...]

    def boundary(self, index: int) -> Fraction:
        ticks = (
            self.pts[index]
            if index < len(self.pts)
            else self.pts[-1] + self.durations[-1]
        )
        return (ticks - self.pts[0]) * self.time_base


def probe_video(path: Path) -> Timeline:
    timestamps: list[int] = []
    last_duration = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("input has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        _reject_hdr(stream.codec_context.color_trc, path)
        if stream.time_base is None or stream.average_rate is None:
            raise ValueError("video must declare a time base and nominal frame rate")
        time_base = Fraction(stream.time_base)
        rate = Fraction(stream.average_rate)
        width, height = stream.width, stream.height
        if width % 2 or height % 2:
            raise ValueError(
                "H.264 yuv420p requires even dimensions; automatic resizing is disabled"
            )
        if stream.sample_aspect_ratio not in (None, Fraction(0), Fraction(1)):
            raise ValueError(
                "non-square pixels require an explicit coordinate transform"
            )
        if int(stream.metadata.get("rotate", "0")) % 360:
            raise ValueError("rotated video requires an explicit coordinate transform")
        for frame in container.decode(stream):
            _reject_hdr(frame.color_trc, path)
            if frame.side_data.get("DISPLAYMATRIX") is not None:
                raise ValueError(
                    "display-matrix video requires an explicit coordinate transform"
                )
            if (frame.width, frame.height) != (width, height):
                raise ValueError("resolution changes within video")
            if frame.pts is None or frame.time_base != time_base:
                raise ValueError("missing PTS or changing video time base")
            if timestamps and frame.pts <= timestamps[-1]:
                raise ValueError("presentation timestamps must be strictly increasing")
            timestamps.append(frame.pts)
            last_duration = frame.duration
    if not timestamps or last_duration <= 0:
        raise ValueError(
            "video must contain frames with a known final display duration"
        )
    durations = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
    durations.append(last_duration)
    return Timeline(width, height, time_base, rate, tuple(timestamps), tuple(durations))


def decode_range(
    path: Path, timeline: Timeline, start: int, stop: int
) -> Iterator[av.VideoFrame]:
    if not 0 <= start < stop <= len(timeline.pts):
        raise ValueError("invalid decode interval")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(
            timeline.pts[start], stream=stream, backward=True, any_frame=False
        )
        index = start
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("decoder returned a frame without PTS")
            if frame.pts < timeline.pts[start]:
                continue
            if index == stop:
                break
            if frame.pts != timeline.pts[index]:
                raise ValueError(f"decode/PTS mismatch at source frame {index}")
            yield frame
            index += 1
        if index != stop:
            raise ValueError(f"decoder ended at {index}, expected {stop}")


def encode_video(
    path: Path,
    *,
    width: int,
    height: int,
    time_base: Fraction,
    rate: Fraction,
    frames: Iterable[tuple[av.VideoFrame, int, int]],
    crf: int,
    preset: str,
) -> None:
    """Encode without a CFR resampler; explicitly preserve each display duration."""
    durations: dict[Fraction, Fraction] = {}
    with av.open(
        str(path), "w", format="mp4", options={"movflags": "+faststart"}
    ) as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.codec_context.thread_count = 2
        stream.options = {"crf": str(crf), "preset": preset}
        # MP4 durations are DTS deltas. With B-frame reordering, the last
        # displayed frame can lose its explicit duration to a later DTS.
        stream.codec_context.max_b_frames = 0

        def mux(packet: av.Packet) -> None:
            if packet.pts is None or packet.time_base is None:
                raise ValueError("encoder produced a packet without presentation time")
            duration = durations[packet.pts * packet.time_base] / packet.time_base
            if duration.denominator != 1:
                raise ValueError("encoder cannot represent source frame duration")
            packet.duration = int(duration)
            output.mux(packet)

        first = True
        for frame, pts, duration in frames:
            _reject_hdr(frame.color_trc, path)
            if first:
                for attribute in COLOR_ATTRIBUTES:
                    setattr(
                        stream.codec_context,
                        attribute,
                        getattr(frame, attribute),
                    )
                first = False
            converted = frame.reformat(width=width, height=height, format="yuv420p")
            converted.pts = pts
            converted.time_base = time_base
            durations[pts * time_base] = duration * time_base
            for packet in stream.encode(converted):
                mux(packet)
        for packet in stream.encode():
            mux(packet)


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
