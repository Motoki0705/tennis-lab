"""JPEG byte shards with a compressed, pickle-free sparse-record index.

The caller owns the dataset schema and atomic publication. This module owns
byte addressing, integrity checks, JPEG decoding and worker-local mmap handles.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, cast

import cv2
import numpy as np
from numpy.typing import NDArray

FORMAT = "jpeg_sparse_records_v1"
_INDEX_KEYS = {
    "shard",
    "offset",
    "length",
    "sha256",
    "record_offset",
    "record_length",
    "records",
    "metadata",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _contained(root: Path, relative: str) -> Path:
    path = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
    ):
        raise ValueError(f"Invalid image-store path: {relative!r}")
    if (
        not path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        or not path.is_file()
    ):
        raise ValueError(f"Image-store path escapes its owner: {relative!r}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_rgb_jpeg(rgb: NDArray[np.uint8], *, quality: int = 95) -> bytes:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("JPEG input must be uint8 RGB [H,W,3].")
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in 1..100.")
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise ValueError("JPEG encoding failed.")
    return encoded.tobytes()


class ImageRecordWriter:
    """Append accepted images and sparse JSON records to a fresh owner."""

    def __init__(self, root: Path, *, shard_bytes: int = 256 * 1024 * 1024) -> None:
        if shard_bytes <= 0:
            raise ValueError("shard_bytes must be positive.")
        self.root = root
        self.shard_bytes = shard_bytes
        (root / "shards").mkdir(parents=True, exist_ok=False)
        self._stream: BinaryIO | None = None
        self._shards: list[str] = []
        self._rows: list[tuple[int, int, int, str]] = []
        self._records: list[bytes] = []
        self._finished = False

    def append(self, jpeg: bytes, record: Mapping[str, object]) -> int:
        if self._finished:
            raise RuntimeError("Image store is already finished.")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("Image store accepts complete JPEG images only.")
        payload = _json_bytes(dict(record))
        if self._stream is None or (
            self._stream.tell() > 0
            and self._stream.tell() + len(jpeg) > self.shard_bytes
        ):
            if self._stream is not None:
                self._stream.close()
            relative = f"shards/images-{len(self._shards):05d}.bin"
            self._stream = (self.root / relative).open("xb")
            self._shards.append(relative)
        offset = self._stream.tell()
        self._stream.write(jpeg)
        self._rows.append(
            (len(self._shards) - 1, offset, len(jpeg), hashlib.sha256(jpeg).hexdigest())
        )
        self._records.append(payload)
        return len(self._rows) - 1

    def finish(self, metadata: Mapping[str, object]) -> dict[str, object]:
        if self._finished or not self._rows:
            raise ValueError("A store must be non-empty and finished exactly once.")
        self.close()
        self._finished = True
        lengths = np.asarray([len(record) for record in self._records], dtype=np.int64)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]))
        index_path = self.root / "index.npz"
        with index_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                shard=np.asarray([row[0] for row in self._rows], dtype=np.int32),
                offset=np.asarray([row[1] for row in self._rows], dtype=np.int64),
                length=np.asarray([row[2] for row in self._rows], dtype=np.int64),
                sha256=np.asarray([row[3] for row in self._rows], dtype="S64"),
                record_offset=offsets,
                record_length=lengths,
                records=np.frombuffer(b"".join(self._records), dtype=np.uint8),
                metadata=np.frombuffer(_json_bytes(dict(metadata)), dtype=np.uint8),
            )
        return {
            "format": FORMAT,
            "count": len(self._rows),
            "index": "index.npz",
            "index_sha256": file_sha256(index_path),
            "shards": [
                {
                    "path": relative,
                    "bytes": (self.root / relative).stat().st_size,
                    "sha256": file_sha256(self.root / relative),
                }
                for relative in self._shards
            ],
        }

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> ImageRecordWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ImageRecordStore:
    """Read only one compressed image per access; never unpickle index data."""

    def __init__(self, root: Path, descriptor: Mapping[str, Any]) -> None:
        if (
            set(descriptor) != {"format", "count", "index", "index_sha256", "shards"}
            or descriptor["format"] != FORMAT
        ):
            raise ValueError("Unknown image-store format or descriptor fields.")
        if type(descriptor["count"]) is not int or descriptor["count"] <= 0:
            raise ValueError("Image-store count must be positive.")
        self.root = root.resolve(strict=True)
        self.descriptor = dict(descriptor)
        index_path = _contained(self.root, descriptor["index"])
        if file_sha256(index_path) != descriptor["index_sha256"]:
            raise ValueError("Image-store index checksum mismatch.")
        with np.load(index_path, allow_pickle=False) as archive:
            if set(archive.files) != _INDEX_KEYS:
                raise ValueError("Image-store index columns changed.")
            self.arrays = {key: archive[key] for key in archive.files}
        for array in self.arrays.values():
            array.setflags(write=False)
        count = descriptor["count"]
        for name, dtype in (
            ("shard", "int32"),
            ("offset", "int64"),
            ("length", "int64"),
            ("record_offset", "int64"),
            ("record_length", "int64"),
            ("sha256", "S64"),
        ):
            if self.arrays[name].shape != (count,) or self.arrays[
                name
            ].dtype != np.dtype(dtype):
                raise ValueError(f"Invalid image-store column: {name}.")
        for name in ("records", "metadata"):
            if self.arrays[name].dtype != np.uint8 or self.arrays[name].ndim != 1:
                raise ValueError(f"Invalid image-store byte column: {name}.")
        self.metadata = json.loads(self.arrays["metadata"].tobytes())
        if not isinstance(self.metadata, dict):
            raise ValueError("Image-store metadata must be an object.")
        shards = descriptor["shards"]
        if not isinstance(shards, list) or not shards:
            raise ValueError("Image store requires a shard inventory.")
        self.paths: list[Path] = []
        for shard in shards:
            if not isinstance(shard, dict) or set(shard) != {"path", "bytes", "sha256"}:
                raise ValueError("Invalid image-store shard record.")
            path = _contained(self.root, shard["path"])
            if path.stat().st_size != shard["bytes"] or path in self.paths:
                raise ValueError("Image-store shard size or identity mismatch.")
            self.paths.append(path)
        next_image_offset = [0] * len(shards)
        next_record_offset = 0
        for row in range(count):
            shard = int(self.arrays["shard"][row])
            offset, length = (
                int(self.arrays["offset"][row]),
                int(self.arrays["length"][row]),
            )
            start, size = (
                int(self.arrays["record_offset"][row]),
                int(self.arrays["record_length"][row]),
            )
            if (
                not 0 <= shard < len(shards)
                or length <= 0
                or offset != next_image_offset[shard]
            ):
                raise ValueError("Image-store image ranges overlap or have gaps.")
            if size <= 0 or start != next_record_offset:
                raise ValueError("Image-store record ranges overlap or have gaps.")
            next_image_offset[shard] += length
            next_record_offset += size
        if next_image_offset != [
            shard["bytes"] for shard in shards
        ] or next_record_offset != len(self.arrays["records"]):
            raise ValueError("Image-store ranges do not cover their payloads exactly.")
        self._maps: dict[int, np.memmap] = {}
        self._pid = os.getpid()

    def __len__(self) -> int:
        return int(self.descriptor["count"])

    def record(self, row: int) -> dict[str, Any]:
        if not 0 <= row < len(self):
            raise IndexError(row)
        start, size = (
            int(self.arrays["record_offset"][row]),
            int(self.arrays["record_length"][row]),
        )
        result = json.loads(self.arrays["records"][start : start + size].tobytes())
        if not isinstance(result, dict):
            raise ValueError("Image-store record must be an object.")
        return result

    def jpeg(self, row: int) -> bytes:
        if not 0 <= row < len(self):
            raise IndexError(row)
        if self._pid != os.getpid():
            self._maps = {}
            self._pid = os.getpid()
        shard = int(self.arrays["shard"][row])
        if shard not in self._maps:
            self._maps[shard] = np.memmap(self.paths[shard], mode="r", dtype=np.uint8)
        start, size = int(self.arrays["offset"][row]), int(self.arrays["length"][row])
        jpeg = self._maps[shard][start : start + size].tobytes()
        if (
            hashlib.sha256(jpeg).hexdigest().encode("ascii")
            != self.arrays["sha256"][row]
        ):
            raise ValueError(f"Image-store JPEG checksum mismatch at row {row}.")
        return jpeg

    def rgb(self, row: int) -> NDArray[np.uint8]:
        image = cv2.imdecode(
            np.frombuffer(self.jpeg(row), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError(f"Invalid JPEG at row {row}.")
        return cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def validate(self, *, decode: bool = True) -> None:
        for path, shard in zip(self.paths, self.descriptor["shards"], strict=True):
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Image-store shard checksum mismatch: {path.name}")
        for row in range(len(self)):
            record = self.record(row)
            if decode:
                image = self.rgb(row)
                if image.shape[:2] != (record["height"], record["width"]):
                    raise ValueError(
                        f"Image-store RGB dimensions disagree at row {row}."
                    )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._maps = {}
        self._pid = os.getpid()
        for array in self.arrays.values():
            array.setflags(write=False)
