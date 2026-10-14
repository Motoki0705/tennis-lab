"""Stream at most one 24-frame contact-sheet page in memory at a time."""

from __future__ import annotations

import math
import os
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    sync_directory,
    verify_reference,
)
from ..contracts import ArtifactRef, content_digest
from ..paths import AnnotationLayout
from ..plan import file_digest
from .contracts import ImagePage, ImageSample, PreviewReceipt


def make_sheets(
    layout: AnnotationLayout,
    source: ArtifactRef,
    output: Path,
    *,
    origin_pts: int,
    time_base: str,
    start: float,
    stop: float,
    step: float,
) -> ArtifactRef:
    if (
        not all(math.isfinite(value) for value in (start, stop, step))
        or not 0 <= start < stop
        or step <= 0
    ):
        raise ValueError("invalid preview sampling interval")
    clock = Fraction(time_base)
    if clock <= 0:
        raise ValueError("source time base must be positive")
    # Resolve outputs through the same fixed root policy as all evidence.
    layout.artifact_path(
        ArtifactRef(
            role="annotation",
            path=output.relative_to(layout.root).as_posix(),
            digest="0" * 64,
            size_bytes=0,
        )
    )
    request_id = content_digest(
        [
            source.model_dump(),
            origin_pts,
            time_base,
            start,
            stop,
            step,
            "384x216+28/24.v1",
        ]
    )
    directory = output / request_id
    source_path = verify_reference(layout, source)
    with artifact_lock(layout, "screen-preview-" + request_id):
        if directory.exists():
            ref = annotation_reference(layout, directory / "preview.json")
            preview = read_document(layout, ref, PreviewReceipt)
            if (
                preview.source,
                preview.origin_pts,
                preview.time_base,
                preview.start,
                preview.stop,
                preview.step,
            ) != (source, origin_pts, time_base, start, stop, step):
                raise ValueError(
                    "existing preview differs from the requested source/interval"
                )
            for page in preview.pages:
                verify_reference(layout, page.image)
            return ref
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="building-", dir=output) as temporary:
            staging = Path(temporary)
            tiles: list[NDArray[np.uint8]] = []
            samples: list[ImageSample] = []
            pages: list[ImagePage] = []

            def flush() -> None:
                if not tiles:
                    return
                sheet: NDArray[np.uint8] = np.zeros(
                    (math.ceil(len(tiles) / 4) * 244, 384 * 4, 3), np.uint8
                )
                for index, tile in enumerate(tiles):
                    row, column = divmod(index, 4)
                    sheet[
                        row * 244 : (row + 1) * 244, column * 384 : (column + 1) * 384
                    ] = tile
                filename = f"page-{len(pages):04d}.jpg"
                path = staging / filename
                if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError(f"could not write screening image {path}")
                ref = ArtifactRef(
                    role="annotation",
                    path=(directory / filename).relative_to(layout.root).as_posix(),
                    digest=file_digest(path),
                    size_bytes=path.stat().st_size,
                )
                pages.append(ImagePage(image=ref, samples=tuple(samples)))
                tiles.clear()
                samples.clear()

            next_time, last_time = start, start
            previous_pts: int | None = None
            with av.open(str(source_path)) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                if stream.time_base != clock:
                    raise ValueError(
                        "source time base differs from the screening request"
                    )
                if start:
                    container.seek(
                        origin_pts + int(Fraction(str(start)) / clock),
                        stream=stream,
                        backward=True,
                    )
                for frame in container.decode(stream):
                    if frame.pts is None or frame.time_base != clock:
                        raise ValueError("source frame lacks its declared PTS/duration")
                    if previous_pts is not None and frame.pts <= previous_pts:
                        raise ValueError("source PTS must increase")
                    previous_pts = frame.pts
                    timestamp = float((frame.pts - origin_pts) * clock)
                    if timestamp >= stop:
                        last_time = timestamp
                        break
                    last_time = timestamp + float(frame.duration * clock)
                    if timestamp + 1e-6 < next_time:
                        continue
                    pixels = frame.to_ndarray(format="bgr24")
                    scale = min(384 / frame.width, 216 / frame.height)
                    preview_pixels = cv2.resize(
                        pixels,
                        (
                            max(1, round(frame.width * scale)),
                            max(1, round(frame.height * scale)),
                        ),
                    )
                    tile: NDArray[np.uint8] = np.zeros((244, 384, 3), np.uint8)
                    tile[
                        28 : 28 + preview_pixels.shape[0], : preview_pixels.shape[1]
                    ] = preview_pixels
                    cv2.putText(
                        tile,
                        f"source t={timestamp:.3f}s",
                        (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    tiles.append(tile)
                    samples.append(ImageSample(source_pts=frame.pts, seconds=timestamp))
                    next_time = (
                        start + (math.floor((timestamp - start) / step) + 1) * step
                    )
                    if len(tiles) == 24:
                        flush()
            flush()
            if not pages:
                raise ValueError(
                    "no real source frame falls inside the requested preview interval"
                )
            if last_time + 1e-6 < stop:
                raise ValueError("source ended before the requested preview interval")
            receipt = PreviewReceipt(
                source=source,
                origin_pts=origin_pts,
                time_base=time_base,
                start=start,
                stop=stop,
                step=step,
                decoded_until=last_time,
                pages=tuple(pages),
            )
            (staging / "preview.json").write_bytes(receipt.canonical_bytes() + b"\n")
            for path in staging.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            sync_directory(staging)
            os.rename(staging, directory)
            sync_directory(output)
        return annotation_reference(layout, directory / "preview.json")
