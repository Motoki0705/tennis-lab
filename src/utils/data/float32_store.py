"""Lossless float32 storage: explicit NPY or versioned byte-shuffled NPZ.

The archive stores four byte planes before DEFLATE. No quantization occurs;
IEEE bits (including signed zero) survive. A SHA-256 checks decoded bytes.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, cast
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

SUFFIX = ".f32.npz"
_SCHEMA = "float32_byte_planes_v1"
_MAX_BYTES = 512 * 1024 * 1024


def write_float32(path: Path, array: NDArray[np.float32]) -> None:
    """Create a new archive; callers own atomic publication of its reference."""
    if not path.name.endswith(SUFFIX) or array.dtype != np.dtype("<f4"):
        raise ValueError("Expected .f32.npz destination and little-endian float32.")
    if (
        not 1 <= array.ndim <= 8
        or array.nbytes > _MAX_BYTES
        or any(n <= 0 for n in array.shape)
    ):
        raise ValueError("Invalid or oversized float32 array.")
    value = np.ascontiguousarray(array)
    raw = value.tobytes()
    planes = value.view(np.uint8).reshape(-1, 4).T.copy()
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            schema=np.array(_SCHEMA),
            shape=np.array(array.shape, dtype=np.int64),
            sha256=np.array(hashlib.sha256(raw).hexdigest()),
            byte_planes=planes,
        )


def read_float32(
    path: Path, *, mmap_mode: Literal["r", "r+", "w+", "c"] | None = None
) -> NDArray[np.float32]:
    """Decode the explicitly named codec; never search for substitute files."""
    if path.suffix == ".npy":
        return cast(
            NDArray[np.float32], np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        )
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"Unknown float32 storage codec: {path}")
    _validate_archive_members(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"schema", "shape", "sha256", "byte_planes"}:
            raise ValueError("Invalid float32 archive fields.")
        if archive["schema"].shape != () or str(archive["schema"].item()) != _SCHEMA:
            raise ValueError("Unknown float32 archive version.")
        shape_array = archive["shape"]
        if (
            shape_array.dtype != np.int64
            or shape_array.ndim != 1
            or not 1 <= len(shape_array) <= 8
        ):
            raise ValueError("Invalid float32 archive shape.")
        shape = tuple(int(n) for n in shape_array)
        count = math.prod(shape)
        if any(n <= 0 for n in shape) or count * 4 > _MAX_BYTES:
            raise ValueError("Oversized float32 archive.")
        planes = archive["byte_planes"]
        if planes.dtype != np.uint8 or planes.shape != (4, count):
            raise ValueError("Invalid float32 byte planes.")
        raw = planes.T.copy().tobytes()
        if archive["sha256"].shape != () or hashlib.sha256(raw).hexdigest() != str(
            archive["sha256"].item()
        ):
            raise ValueError("Float32 archive checksum mismatch.")
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()


def _validate_archive_members(path: Path) -> None:
    with ZipFile(path) as zipped:
        members = zipped.infolist()
        expected = {"schema.npy", "shape.npy", "sha256.npy", "byte_planes.npy"}
        if len(members) != 4 or {member.filename for member in members} != expected:
            raise ValueError("Invalid float32 archive fields.")
        for member in members:
            limit = _MAX_BYTES + 1024 if member.filename == "byte_planes.npy" else 4096
            if member.file_size > limit:
                raise ValueError("Oversized float32 archive member.")


def inspect_float32(path: Path) -> tuple[np.dtype, tuple[int, ...]]:
    """Inspect array headers without decompressing image data or checking pixels."""
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        return array.dtype, array.shape
    if not path.name.endswith(SUFFIX):
        raise ValueError(f"Unknown float32 storage codec: {path}")
    _validate_archive_members(path)
    with np.load(path, allow_pickle=False) as archive:
        if archive["schema"].shape != () or archive["schema"].item() != _SCHEMA:
            raise ValueError("Unknown float32 archive version.")
        shape_array = archive["shape"]
        if (
            shape_array.dtype != np.int64
            or shape_array.ndim != 1
            or not 1 <= len(shape_array) <= 8
        ):
            raise ValueError("Invalid float32 archive shape.")
        shape = tuple(int(n) for n in shape_array)
        if any(n <= 0 for n in shape) or math.prod(shape) * 4 > _MAX_BYTES:
            raise ValueError("Oversized float32 archive.")
    with ZipFile(path) as zipped, zipped.open("byte_planes.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("Unexpected byte-plane NPY version.")
        planes_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if dtype != np.uint8 or fortran or planes_shape != (4, math.prod(shape)):
            raise ValueError("Invalid float32 byte-plane header.")
        if (
            zipped.getinfo("byte_planes.npy").file_size
            != stream.tell() + math.prod(shape) * 4
        ):
            raise ValueError("Invalid byte-plane member size.")
    return np.dtype("<f4"), shape
