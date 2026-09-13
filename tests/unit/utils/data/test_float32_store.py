from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.data.float32_store import read_float32, write_float32


def test_bitwise_roundtrip_and_noncontiguous(tmp_path: Path) -> None:
    bits = np.array(
        [0, 0x80000000, 0x7FC00001, 0x7F800000, 0x00000001, 0x3F800000], dtype=np.uint32
    )
    value = bits.view(np.float32).reshape(2, 3).T
    path = tmp_path / "value.f32.npz"
    write_float32(path, value)
    assert read_float32(path).tobytes() == value.tobytes()
    with pytest.raises(FileExistsError):
        write_float32(path, value)


def test_checksum_and_unknown_codec_fail(tmp_path: Path) -> None:
    path = tmp_path / "value.f32.npz"
    write_float32(path, np.ones((2, 3), dtype=np.float32))
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["byte_planes"][0, 0] ^= 1
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        read_float32(path)
    with pytest.raises(ValueError, match="codec"):
        read_float32(tmp_path / "value.zip")


def test_npy_and_truncated_archive(tmp_path: Path) -> None:
    value: NDArray[np.float32] = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    path = tmp_path / "value.npy"
    np.save(path, value)
    np.testing.assert_array_equal(read_float32(path, mmap_mode="r"), value)
    compressed = tmp_path / "value.f32.npz"
    write_float32(compressed, value)
    compressed.write_bytes(compressed.read_bytes()[:30])
    with pytest.raises(BadZipFile):
        read_float32(compressed)


def test_header_scan_and_shape_tampering(tmp_path: Path) -> None:
    from src.utils.data.float32_store import inspect_float32

    path = tmp_path / "value.f32.npz"
    write_float32(path, np.zeros((3, 4, 1), dtype=np.float32))
    assert inspect_float32(path) == (np.dtype("float32"), (3, 4, 1))
    with np.load(path) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["shape"] = np.array([3, 5, 1], dtype=np.int64)
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="header"):
        inspect_float32(path)


def test_writer_rejects_shape_it_cannot_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or oversized"):
        write_float32(
            tmp_path / "too_many_axes.f32.npz", np.zeros((1,) * 9, dtype=np.float32)
        )
