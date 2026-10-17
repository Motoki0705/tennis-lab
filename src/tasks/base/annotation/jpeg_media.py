"""Native JPEG frames shared by annotation and published training datasets."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .contracts import JpegFrame, JpegStorage

if TYPE_CHECKING:
    from .media import Timeline


def decode_jpeg(payload: bytes, width: int, height: int) -> NDArray[np.uint8]:
    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None or pixels.shape != (height, width, 3):
        raise ValueError("JPEG does not decode to its declared native dimensions")
    result: NDArray[np.uint8] = pixels
    return result


@dataclass(frozen=True)
class JpegArchive:
    path: Path
    storage: JpegStorage
    width: int
    height: int

    def read_range(self, start: int, stop: int) -> Iterator[bytes]:
        if not 0 <= start < stop <= len(self.storage.frames):
            raise ValueError("JPEG read lies outside the real frame range")
        with self.path.open("rb") as handle:
            for row in self.storage.frames[start:stop]:
                handle.seek(row.offset)
                payload = handle.read(row.length)
                if (
                    len(payload) != row.length
                    or hashlib.sha256(payload).hexdigest() != row.digest
                ):
                    raise ValueError(
                        "JPEG frame bytes differ from the preparation manifest"
                    )
                yield payload

    def iter_frames(self) -> Iterator[bytes]:
        yield from self.read_range(0, len(self.storage.frames))


def write_jpeg_archive(
    path: Path, source: Path, timeline: Timeline, start: int, stop: int
) -> JpegStorage:
    """Encode each real source frame once; no intermediate clip video or resize."""
    from .media import decode_range

    rows = []
    offset = 0
    with path.open("xb") as output:
        for frame in decode_range(source, timeline, start, stop):
            pixels = frame.to_ndarray(format="bgr24")
            if pixels.shape != (timeline.height, timeline.width, 3):
                raise ValueError(
                    "source frame dimensions changed during JPEG preparation"
                )
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError("native JPEG encoding failed")
            payload = encoded.tobytes()
            output.write(payload)
            rows.append(
                JpegFrame(
                    offset=offset,
                    length=len(payload),
                    digest=hashlib.sha256(payload).hexdigest(),
                )
            )
            offset += len(payload)
        output.flush()
        os.fsync(output.fileno())
    if len(rows) != stop - start:
        raise ValueError("JPEG preparation did not cover all source frames")
    return JpegStorage(frames=tuple(rows))
