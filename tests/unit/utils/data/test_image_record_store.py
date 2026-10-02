"""Image sharding must retain byte identity, random access and worker safety."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.data.image_record_store import (
    ImageRecordStore,
    ImageRecordWriter,
    encode_rgb_jpeg,
)


def test_rgb_channel_order_survives_encoding_and_decoding(tmp_path: Path) -> None:
    rgb: NDArray[np.uint8] = np.empty((16, 16, 3), dtype=np.uint8)
    rgb[:] = (220, 10, 30)
    with ImageRecordWriter(tmp_path) as writer:
        writer.append(encode_rgb_jpeg(rgb), {"width": 16, "height": 16})
        descriptor = writer.finish({})
    restored = ImageRecordStore(tmp_path, descriptor).rgb(0)
    np.testing.assert_allclose(restored.mean(axis=(0, 1)), (220, 10, 30), atol=3)


def test_sharded_random_access_preserves_jpeg_and_sparse_labels(tmp_path: Path) -> None:
    images = [
        encode_rgb_jpeg(np.full((12, 16, 3), color, np.uint8))
        for color in (20, 80, 180)
    ]
    with ImageRecordWriter(tmp_path, shard_bytes=100) as writer:
        for index, image in enumerate(images):
            writer.append(
                image,
                {
                    "id": f"sample-{index}",
                    "width": 16,
                    "height": 12,
                    "kp14": [[-2.5, 50.25]] * 14,
                },
            )
        descriptor = writer.finish({"schema": "test_v1"})
    store = ImageRecordStore(tmp_path, descriptor)
    store.validate()
    assert len(store.paths) == 3
    for index in (2, 0, 1):
        assert store.jpeg(index) == images[index]
        assert store.record(index)["kp14"][0] == [-2.5, 50.25]
        assert store.rgb(index).shape == (12, 16, 3)
    restored = pickle.loads(pickle.dumps(store))
    assert not restored._maps
    assert restored.jpeg(1) == images[1]


def test_corrupt_jpeg_is_rejected_before_decode(tmp_path: Path) -> None:
    with ImageRecordWriter(tmp_path) as writer:
        writer.append(
            encode_rgb_jpeg(np.zeros((8, 8, 3), np.uint8)), {"width": 8, "height": 8}
        )
        descriptor = writer.finish({})
    store = ImageRecordStore(tmp_path, descriptor)
    path = store.paths[0]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.rgb(0)


def test_index_tampering_and_traversal_are_rejected(tmp_path: Path) -> None:
    with ImageRecordWriter(tmp_path) as writer:
        writer.append(
            encode_rgb_jpeg(np.zeros((8, 8, 3), np.uint8)), {"width": 8, "height": 8}
        )
        descriptor = writer.finish({})
    with pytest.raises(ValueError, match="Invalid image-store path"):
        ImageRecordStore(tmp_path, {**descriptor, "index": "../index.npz"})
    with (tmp_path / "index.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ImageRecordStore(tmp_path, descriptor)
