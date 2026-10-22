"""Point Cache 2 (``.pc2``) vertex-animation files for Blender's Mesh Cache modifier.

Layout (little endian): 12-byte signature ``b"POINTCACHE2\\0"``, int32 version
(1), int32 point count, float32 start frame, float32 sample rate, int32 sample
count, then ``samples * points * 3`` float32 coordinates.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SIGNATURE = b"POINTCACHE2\0"
_HEADER = struct.Struct("<12siiffi")


def write_pc2(
    path: Path,
    frames: NDArray[np.floating],
    *,
    start_frame: float,
    sample_rate: float = 1.0,
) -> None:
    """Write ``(samples, points, 3)`` finite vertex positions as a PC2 file."""
    data = np.asarray(frames, dtype="<f4")
    if data.ndim != 3 or data.shape[2] != 3 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"PC2 frames must have shape (samples, points, 3), got {data.shape}."
        )
    if not np.isfinite(data).all():
        raise ValueError("PC2 frames must be finite.")
    if sample_rate <= 0.0:
        raise ValueError("PC2 sample_rate must be positive.")
    samples, points, _ = data.shape
    with path.open("wb") as handle:
        handle.write(
            _HEADER.pack(_SIGNATURE, 1, points, start_frame, sample_rate, samples)
        )
        handle.write(np.ascontiguousarray(data).tobytes())


def read_pc2(path: Path) -> tuple[NDArray[np.float32], float, float]:
    """Return ``(frames, start_frame, sample_rate)`` from a PC2 file."""
    raw = path.read_bytes()
    signature, version, points, start, rate, samples = _HEADER.unpack_from(raw)
    if signature != _SIGNATURE or version != 1:
        raise ValueError(f"{path} is not a version-1 PC2 file.")
    expected = _HEADER.size + samples * points * 3 * 4
    if len(raw) != expected:
        raise ValueError(f"{path} has {len(raw)} bytes, expected {expected}.")
    frames = np.frombuffer(raw, dtype="<f4", offset=_HEADER.size)
    return (
        frames.reshape(samples, points, 3).astype(np.float32),
        float(start),
        float(rate),
    )


__all__ = ["read_pc2", "write_pc2"]
