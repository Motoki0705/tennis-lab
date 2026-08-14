"""Attempt-local compact render storage shared by synthetic dataset domains.

The canonical scene pipeline renders static camera backgrounds once and stores
only visible foreground changes for timeline samples.  This module owns that
synthetic-dataset contract; task labels and domain manifests remain with Court,
BLCS, and PLCS.
"""

from __future__ import annotations

import json
import math
import os
import resource
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderArrays,
    NHTRenderResult,
)

BACKGROUND_STORE_SCHEMA = "shared_render_background_store_v1"
FOREGROUND_CHUNK_SCHEMA = "foreground_delta_chunk_v1"
PERFORMANCE_SCHEMA = "dataset_performance_metrics_v1"


@dataclass(frozen=True, slots=True, order=True)
class RenderSampleKey:
    """Stable logical identity for one global-frame/camera render."""

    frame_index: int
    camera_id: str

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        _identifier(self.camera_id, name="camera_id")


@dataclass(frozen=True, slots=True)
class ForegroundDelta:
    """Sparse final pixel values over one validated static background."""

    key: RenderSampleKey
    pixel_indices: NDArray[np.int32]
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    instance_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        if not isinstance(self.key, RenderSampleKey):
            raise TypeError("key must be a RenderSampleKey.")
        pixels = _array(self.pixel_indices, np.int32, (None,), name="pixel_indices")
        count = len(pixels)
        rgb = _array(self.rgb, np.float32, (count, 3), name="rgb")
        alpha = _array(self.alpha, np.float32, (count,), name="alpha")
        depth = _array(self.depth, np.float32, (count,), name="depth")
        instances = _array(
            self.instance_ids,
            np.int32,
            (count,),
            name="instance_ids",
        )
        if count and (np.any(pixels < 0) or np.any(np.diff(pixels) <= 0)):
            raise ValueError("pixel_indices must be sorted, unique, and non-negative.")
        if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("Foreground delta RGB must be finite and in [0,1].")
        if (
            not np.isfinite(alpha).all()
            or np.any(alpha < 0.0)
            or np.any(alpha > 1.0)
        ):
            raise ValueError("Foreground delta alpha must be finite and in [0,1].")
        if not np.isfinite(depth).all() or np.any(depth <= 0.0):
            raise ValueError("Foreground delta depth must be finite and positive.")
        if np.any(instances <= 0):
            raise ValueError("Foreground delta instance IDs must be positive.")
        for value in (pixels, rgb, alpha, depth, instances):
            value.setflags(write=False)
        object.__setattr__(self, "pixel_indices", pixels)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "instance_ids", instances)

    @property
    def visible_instance_counts(self) -> dict[int, int]:
        """Return exact sparse-pixel counts for every visible instance."""
        values, counts = np.unique(self.instance_ids, return_counts=True)
        return {
            int(instance_id): int(count)
            for instance_id, count in zip(values, counts, strict=True)
        }


@dataclass(frozen=True, slots=True)
class ForegroundDeltaBatch:
    """One ordered render chunk submitted to the atomic chunk writer."""

    chunk_id: str
    deltas: tuple[ForegroundDelta, ...]
    metadata: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        chunk_id = _identifier(self.chunk_id, name="chunk_id")
        deltas = tuple(self.deltas)
        metadata = tuple(dict(value) for value in self.metadata)
        if not deltas or len(deltas) != len(metadata):
            raise ValueError("Chunk deltas and metadata must be non-empty and aligned.")
        if any(not isinstance(value, ForegroundDelta) for value in deltas):
            raise TypeError("Chunk deltas must contain ForegroundDelta values.")
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class BackgroundArrays:
    """One in-memory metric background loaded once by a render session."""

    camera_id: str
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]

    def __post_init__(self) -> None:
        camera_id = _identifier(self.camera_id, name="camera_id")
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Background RGB must have shape [H,W,3].")
        height, width = rgb.shape[:2]
        rgb = _array(rgb, np.float32, (height, width, 3), name="background.rgb")
        alpha = _array(
            self.alpha,
            np.float32,
            (height, width, 1),
            name="background.alpha",
        )
        depth = _array(
            self.depth,
            np.float32,
            (height, width, 1),
            name="background.depth",
        )
        if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("Background RGB must be finite and in [0,1].")
        if (
            not np.isfinite(alpha).all()
            or np.any(alpha < 0.0)
            or np.any(alpha > 1.0)
        ):
            raise ValueError("Background alpha must be finite and in [0,1].")
        if not np.isfinite(depth).all() or np.any(depth < 0.0):
            raise ValueError("Background depth must be finite and non-negative.")
        for value in (rgb, alpha, depth):
            value.setflags(write=False)
        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "depth", depth)

    @property
    def width(self) -> int:
        """Return image width."""
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        """Return image height."""
        return int(self.rgb.shape[0])

    @classmethod
    def from_validated_nht(
        cls,
        camera_id: str,
        arrays: NHTRenderArrays,
        *,
        nht_scene_units_per_metre: float,
    ) -> BackgroundArrays:
        """Transfer one already-scanned public result without rescanning payloads."""
        if not isinstance(arrays, NHTRenderArrays):
            raise TypeError("Background transfer requires validated NHT arrays.")
        value = object.__new__(cls)
        object.__setattr__(value, "camera_id", _identifier(camera_id, name="camera_id"))
        object.__setattr__(value, "rgb", arrays.rgb)
        object.__setattr__(value, "alpha", arrays.alpha)
        object.__setattr__(
            value,
            "depth",
            arrays.metric_depth(
                nht_scene_units_per_metre=nht_scene_units_per_metre,
            ),
        )
        return value


@dataclass(frozen=True, slots=True)
class LogicalRenderSample:
    """Materialized logical arrays reconstructed from background plus delta."""

    key: RenderSampleKey
    rgb: NDArray[np.float32]
    alpha: NDArray[np.float32]
    depth: NDArray[np.float32]
    instance_ids: NDArray[np.int32]


class SharedBackgroundStore:
    """Canonical static-background storage with process-local validated caching."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._records = self._load_manifest()
        self._cache: dict[str, BackgroundArrays] = {}
        self._cache_misses = 0

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        rendered: NHTRenderResult,
        nht_scene_units_per_metre: float,
    ) -> SharedBackgroundStore:
        """Move one validated NHT result into metric, non-duplicated storage."""
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"Background store already exists: {root}")
        root.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, object]] = []
        transferred: dict[str, BackgroundArrays] = {}
        seen: set[str] = set()
        for record in rendered.records:
            if record.camera_id in seen:
                raise ValueError(f"Duplicate background camera: {record.camera_id!r}.")
            seen.add(record.camera_id)
            camera_root = root / record.camera_id
            camera_root.mkdir(parents=False, exist_ok=False)
            rgb_path = camera_root / "rgb.npy"
            alpha_path = camera_root / "alpha.npy"
            depth_path = camera_root / "depth-metric.npy"
            arrays = record.load_arrays()
            background = BackgroundArrays.from_validated_nht(
                record.camera_id,
                arrays,
                nht_scene_units_per_metre=nht_scene_units_per_metre,
            )
            record.rgb_path.replace(rgb_path)
            record.alpha_path.replace(alpha_path)
            np.save(depth_path, background.depth, allow_pickle=False)
            transferred[record.camera_id] = background
            records.append(
                {
                    "camera_id": record.camera_id,
                    "width": record.width,
                    "height": record.height,
                    "rgb": _relative(root, rgb_path),
                    "alpha": _relative(root, alpha_path),
                    "depth": _relative(root, depth_path),
                }
            )
        _write_json_atomic(
            root / "backgrounds.json",
            {
                "schema": BACKGROUND_STORE_SCHEMA,
                "scene_id": rendered.scene_id,
                "depth_coordinate_space": "metric_scene_metres",
                "records": records,
            },
        )
        store = cls(root)
        store._cache.update(transferred)
        store._cache_misses = len(transferred)
        return store

    @property
    def camera_ids(self) -> tuple[str, ...]:
        """Return exact manifest camera order."""
        return tuple(self._records)

    @property
    def cache_misses(self) -> int:
        """Return the number of camera payloads loaded by this session."""
        return self._cache_misses

    def load(self, camera_id: str) -> BackgroundArrays:
        """Load and validate one camera exactly once for this process."""
        if camera_id in self._cache:
            return self._cache[camera_id]
        try:
            record = self._records[camera_id]
        except KeyError as error:
            raise KeyError(f"Unknown shared background camera: {camera_id!r}.") from error
        value = BackgroundArrays(
            camera_id=camera_id,
            rgb=np.load(record["rgb"], allow_pickle=False),
            alpha=np.load(record["alpha"], allow_pickle=False),
            depth=np.load(record["depth"], allow_pickle=False),
        )
        if (value.width, value.height) != (record["width"], record["height"]):
            raise ValueError("Background dimensions disagree with the manifest.")
        self._cache[camera_id] = value
        self._cache_misses += 1
        return value

    def validate_all(self) -> None:
        """Perform one complete semantic scan for every stored camera."""
        for camera_id in self.camera_ids:
            self.load(camera_id)

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        manifest = self.root / "backgrounds.json"
        raw = _mapping(_load_json(manifest), name="background store")
        _exact_keys(
            raw,
            {"schema", "scene_id", "depth_coordinate_space", "records"},
            name="background store",
        )
        if raw["schema"] != BACKGROUND_STORE_SCHEMA:
            raise ValueError("Unsupported shared background schema.")
        if raw["depth_coordinate_space"] != "metric_scene_metres":
            raise ValueError("Shared background depth is not metric.")
        _identifier(raw["scene_id"], name="scene_id")
        records_raw = _sequence(raw["records"], name="background records")
        result: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(records_raw):
            record = _mapping(value, name=f"background records[{index}]")
            _exact_keys(
                record,
                {"camera_id", "width", "height", "rgb", "alpha", "depth"},
                name=f"background records[{index}]",
            )
            camera_id = _identifier(record["camera_id"], name="camera_id")
            if camera_id in result:
                raise ValueError(f"Duplicate shared background camera: {camera_id!r}.")
            width = _positive_integer(record["width"], name="width")
            height = _positive_integer(record["height"], name="height")
            result[camera_id] = {
                "width": width,
                "height": height,
                "rgb": _contained_file(self.root, record["rgb"], name="rgb"),
                "alpha": _contained_file(self.root, record["alpha"], name="alpha"),
                "depth": _contained_file(self.root, record["depth"], name="depth"),
            }
        if not result:
            raise ValueError("Shared background store must contain at least one camera.")
        return result


class RenderSession:
    """Attempt-local owner of NHT invocation and background cache evidence."""

    def __init__(
        self,
        *,
        domain: str,
        attempt_token: str,
        execution_device: str,
    ) -> None:
        self.domain = _identifier(domain, name="domain")
        self.attempt_token = _identifier(attempt_token, name="attempt_token")
        self.execution_device = _identifier(
            execution_device,
            name="execution_device",
        )
        self._nht_invocations = 0
        self._stores: dict[str, SharedBackgroundStore] = {}

    @property
    def nht_invocations(self) -> int:
        """Return NHT calls observed by this exact stage attempt."""
        return self._nht_invocations

    @property
    def background_cache_misses(self) -> int:
        """Return camera payload loads across all registered stores."""
        return sum(store.cache_misses for store in self._stores.values())

    def note_nht_invocation(self) -> None:
        """Record one public ``nht-render`` subprocess invocation."""
        self._nht_invocations += 1

    def create_background_store(
        self,
        store_id: str,
        root: Path,
        *,
        rendered: NHTRenderResult,
        nht_scene_units_per_metre: float,
        expected_camera_ids: Sequence[str],
    ) -> SharedBackgroundStore:
        """Register one render result and validate its exact camera inventory."""
        store_id = _identifier(store_id, name="store_id")
        if store_id in self._stores:
            raise ValueError(f"Duplicate render-session store: {store_id!r}.")
        expected = tuple(
            _identifier(value, name="camera_id") for value in expected_camera_ids
        )
        actual = tuple(record.camera_id for record in rendered.records)
        if not expected or actual != expected:
            raise ValueError(
                "Rendered background camera inventory differs from its request."
            )
        store = SharedBackgroundStore.create(
            root,
            rendered=rendered,
            nht_scene_units_per_metre=nht_scene_units_per_metre,
        )
        if store.camera_ids != expected:
            raise ValueError("Stored background camera order changed during publication.")
        self._stores[store_id] = store
        return store

    def background(self, store_id: str, camera_id: str) -> BackgroundArrays:
        """Return one cached metric background without a silent store fallback."""
        store_id = _identifier(store_id, name="store_id")
        try:
            store = self._stores[store_id]
        except KeyError as error:
            raise KeyError(f"Unknown render-session store: {store_id!r}.") from error
        return store.load(camera_id)


@dataclass(frozen=True, slots=True)
class ValidatedChunk:
    """One fully validated chunk and its exact logical sample keys."""

    chunk_id: str
    attempt_token: str
    keys: tuple[RenderSampleKey, ...]
    directory: Path
    pixel_count: int
    byte_count: int


class ChunkWriter:
    """Write one compact foreground chunk and completion marker atomically."""

    def __init__(
        self,
        root: Path,
        *,
        attempt_token: str,
        camera_ids: Sequence[str],
        width: int,
        height: int,
    ) -> None:
        self.root = root
        self.attempt_token = _identifier(attempt_token, name="attempt_token")
        self.camera_ids = tuple(
            _identifier(value, name="camera_id") for value in camera_ids
        )
        if not self.camera_ids or len(self.camera_ids) != len(set(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique.")
        self.width = _positive_integer(width, name="width")
        self.height = _positive_integer(height, name="height")
        self.root.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        batch: ForegroundDeltaBatch,
    ) -> ChunkReader:
        """Write one whole chunk and return its unscanned retained reader."""
        if not isinstance(batch, ForegroundDeltaBatch):
            raise TypeError("ChunkWriter.write requires a ForegroundDeltaBatch.")
        chunk_id = batch.chunk_id
        delta_tuple = batch.deltas
        metadata_tuple = batch.metadata
        keys = tuple(delta.key for delta in delta_tuple)
        camera_index = {camera_id: index for index, camera_id in enumerate(self.camera_ids)}
        if any(key.camera_id not in camera_index for key in keys):
            raise ValueError("Chunk references an unknown camera.")
        expected_order = tuple(
            sorted(
                keys,
                key=lambda key: (key.frame_index, camera_index[key.camera_id]),
            )
        )
        if keys != expected_order or len(keys) != len(set(keys)):
            raise ValueError(
                "Chunk sample keys must follow frame/configured-camera order and be unique."
            )
        maximum_pixel = self.width * self.height
        if any(
            len(delta.pixel_indices)
            and int(delta.pixel_indices[-1]) >= maximum_pixel
            for delta in delta_tuple
        ):
            raise ValueError("Foreground delta pixel index exceeds the image extent.")
        chunk = self.root / chunk_id
        if chunk.exists() or chunk.is_symlink():
            raise FileExistsError(f"Chunk already exists: {chunk}")
        chunk.mkdir(parents=False, exist_ok=False)
        offsets: NDArray[np.int64] = np.zeros(
            len(delta_tuple) + 1,
            dtype=np.int64,
        )
        offsets[1:] = np.cumsum(
            np.asarray([len(value.pixel_indices) for value in delta_tuple], dtype=np.int64)
        )
        arrays: dict[str, NDArray[Any]] = {
            "frame_indices": np.asarray([key.frame_index for key in keys], dtype=np.int64),
            "camera_indices": np.asarray(
                [camera_index[key.camera_id] for key in keys], dtype=np.int32
            ),
            "offsets": offsets,
            "pixel_indices": _concatenate(delta_tuple, "pixel_indices", np.int32),
            "rgb": _concatenate(delta_tuple, "rgb", np.float32, trailing=(3,)),
            "alpha": _concatenate(delta_tuple, "alpha", np.float32),
            "depth": _concatenate(delta_tuple, "depth", np.float32),
            "instance_ids": _concatenate(delta_tuple, "instance_ids", np.int32),
        }
        arrays_path = chunk / "foreground.npz"
        temporary_arrays = chunk / "foreground.npz.tmp"
        with temporary_arrays.open("wb") as handle:
            np.savez(
                handle,
                frame_indices=arrays["frame_indices"],
                camera_indices=arrays["camera_indices"],
                offsets=arrays["offsets"],
                pixel_indices=arrays["pixel_indices"],
                rgb=arrays["rgb"],
                alpha=arrays["alpha"],
                depth=arrays["depth"],
                instance_ids=arrays["instance_ids"],
            )
        temporary_arrays.replace(arrays_path)
        _write_json_atomic(
            chunk / "metadata.json",
            {
                "records": metadata_tuple,
            },
        )
        byte_count = arrays_path.stat().st_size + (chunk / "metadata.json").stat().st_size
        _write_json_atomic(
            chunk / "chunk.json",
            {
                "schema": FOREGROUND_CHUNK_SCHEMA,
                "chunk_id": chunk_id,
                "attempt_token": self.attempt_token,
                "camera_ids": self.camera_ids,
                "width": self.width,
                "height": self.height,
                "sample_count": len(keys),
                "pixel_count": int(offsets[-1]),
                "byte_count": byte_count,
                "arrays": "foreground.npz",
                "metadata": "metadata.json",
            },
        )
        return ChunkReader(chunk)


class ChunkReader:
    """Strict reader for one complete compact foreground chunk."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._marker: dict[str, Any] | None = None
        self._arrays: dict[str, NDArray[Any]] | None = None
        self._metadata: tuple[dict[str, object], ...] | None = None
        self._validated: ValidatedChunk | None = None

    def validate(self, *, expected_attempt_token: str | None = None) -> ValidatedChunk:
        """Reopen and validate every compact array and metadata record."""
        if self._validated is not None:
            if (
                expected_attempt_token is not None
                and self._validated.attempt_token != expected_attempt_token
            ):
                raise ValueError("Foreground chunk belongs to another stage attempt.")
            return self._validated
        marker = self._read_marker()
        if (
            expected_attempt_token is not None
            and marker["attempt_token"] != expected_attempt_token
        ):
            raise ValueError("Foreground chunk belongs to another stage attempt.")
        arrays = self._read_arrays(marker)
        metadata = self._read_metadata(marker)
        sample_count = marker["sample_count"]
        if len(metadata) != sample_count:
            raise ValueError("Foreground chunk metadata count is incomplete.")
        offsets = arrays["offsets"]
        if offsets.shape != (sample_count + 1,) or offsets[0] != 0:
            raise ValueError("Foreground chunk offsets have the wrong shape/origin.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("Foreground chunk offsets are not monotonic.")
        pixel_count = int(offsets[-1])
        if pixel_count != marker["pixel_count"]:
            raise ValueError("Foreground chunk pixel count disagrees with its marker.")
        expected_shapes = {
            "frame_indices": (sample_count,),
            "camera_indices": (sample_count,),
            "pixel_indices": (pixel_count,),
            "rgb": (pixel_count, 3),
            "alpha": (pixel_count,),
            "depth": (pixel_count,),
            "instance_ids": (pixel_count,),
        }
        expected_dtypes: dict[str, np.dtype[np.generic]] = {
            "frame_indices": np.dtype(np.int64),
            "camera_indices": np.dtype(np.int32),
            "offsets": np.dtype(np.int64),
            "pixel_indices": np.dtype(np.int32),
            "rgb": np.dtype(np.float32),
            "alpha": np.dtype(np.float32),
            "depth": np.dtype(np.float32),
            "instance_ids": np.dtype(np.int32),
        }
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise TypeError(f"Foreground chunk {name} has dtype {arrays[name].dtype}.")
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Foreground chunk {name} has shape {arrays[name].shape}.")
        camera_ids = marker["camera_ids"]
        camera_indices = arrays["camera_indices"]
        if np.any(camera_indices < 0) or np.any(camera_indices >= len(camera_ids)):
            raise ValueError("Foreground chunk camera index is out of range.")
        keys = tuple(
            RenderSampleKey(int(frame), camera_ids[int(camera)])
            for frame, camera in zip(
                arrays["frame_indices"], camera_indices, strict=True
            )
        )
        camera_index_by_id = {
            camera_id: index for index, camera_id in enumerate(camera_ids)
        }
        expected_order = tuple(
            sorted(
                keys,
                key=lambda key: (
                    key.frame_index,
                    camera_index_by_id[key.camera_id],
                ),
            )
        )
        if keys != expected_order or len(keys) != len(set(keys)):
            raise ValueError(
                "Foreground chunk keys do not follow configured camera order."
            )
        maximum_pixel = marker["width"] * marker["height"]
        for index, key in enumerate(keys):
            start = int(offsets[index])
            stop = int(offsets[index + 1])
            ForegroundDelta(
                key=key,
                pixel_indices=arrays["pixel_indices"][start:stop],
                rgb=arrays["rgb"][start:stop],
                alpha=arrays["alpha"][start:stop],
                depth=arrays["depth"][start:stop],
                instance_ids=arrays["instance_ids"][start:stop],
            )
            if stop > start and int(arrays["pixel_indices"][stop - 1]) >= maximum_pixel:
                raise ValueError("Foreground chunk pixel index exceeds image extent.")
        actual_bytes = sum(
            path.stat().st_size
            for path in (
                self.directory / "foreground.npz",
                self.directory / "metadata.json",
            )
        )
        if actual_bytes != marker["byte_count"]:
            raise ValueError("Foreground chunk byte count disagrees with its marker.")
        self._marker = marker
        self._arrays = arrays
        self._metadata = metadata
        result = ValidatedChunk(
            chunk_id=marker["chunk_id"],
            attempt_token=marker["attempt_token"],
            keys=keys,
            directory=self.directory,
            pixel_count=pixel_count,
            byte_count=actual_bytes,
        )
        self._validated = result
        return result

    def deltas(self) -> tuple[ForegroundDelta, ...]:
        """Return zero-copy read-only views after strict validation."""
        validated = self.validate()
        assert self._arrays is not None
        arrays = self._arrays
        offsets = arrays["offsets"]
        return tuple(
            ForegroundDelta(
                key=key,
                pixel_indices=arrays["pixel_indices"][
                    int(offsets[index]) : int(offsets[index + 1])
                ],
                rgb=arrays["rgb"][int(offsets[index]) : int(offsets[index + 1])],
                alpha=arrays["alpha"][int(offsets[index]) : int(offsets[index + 1])],
                depth=arrays["depth"][int(offsets[index]) : int(offsets[index + 1])],
                instance_ids=arrays["instance_ids"][
                    int(offsets[index]) : int(offsets[index + 1])
                ],
            )
            for index, key in enumerate(validated.keys)
        )

    def metadata(self) -> tuple[dict[str, object], ...]:
        """Return chunk-level metadata aligned with :meth:`deltas`."""
        self.validate()
        assert self._metadata is not None
        return self._metadata

    def _read_marker(self) -> dict[str, Any]:
        raw = _mapping(_load_json(self.directory / "chunk.json"), name="chunk marker")
        _exact_keys(
            raw,
            {
                "schema",
                "chunk_id",
                "attempt_token",
                "camera_ids",
                "width",
                "height",
                "sample_count",
                "pixel_count",
                "byte_count",
                "arrays",
                "metadata",
            },
            name="chunk marker",
        )
        if raw["schema"] != FOREGROUND_CHUNK_SCHEMA:
            raise ValueError("Unsupported foreground chunk schema.")
        chunk_id = _identifier(raw["chunk_id"], name="chunk_id")
        if self.directory.name != chunk_id:
            raise ValueError("Foreground chunk directory disagrees with chunk_id.")
        camera_ids = tuple(
            _identifier(value, name="camera_id")
            for value in _sequence(raw["camera_ids"], name="camera_ids")
        )
        if not camera_ids or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("Foreground chunk camera IDs must be non-empty and unique.")
        return {
            **raw,
            "chunk_id": chunk_id,
            "attempt_token": _identifier(raw["attempt_token"], name="attempt_token"),
            "camera_ids": camera_ids,
            "width": _positive_integer(raw["width"], name="width"),
            "height": _positive_integer(raw["height"], name="height"),
            "sample_count": _nonnegative_integer(raw["sample_count"], name="sample_count"),
            "pixel_count": _nonnegative_integer(raw["pixel_count"], name="pixel_count"),
            "byte_count": _nonnegative_integer(raw["byte_count"], name="byte_count"),
            "arrays": _contained_file(self.directory, raw["arrays"], name="arrays"),
            "metadata": _contained_file(self.directory, raw["metadata"], name="metadata"),
        }

    def _read_arrays(self, marker: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
        with np.load(marker["arrays"], allow_pickle=False) as archive:
            expected = {
                "frame_indices",
                "camera_indices",
                "offsets",
                "pixel_indices",
                "rgb",
                "alpha",
                "depth",
                "instance_ids",
            }
            if set(archive.files) != expected:
                raise ValueError("Foreground chunk array inventory is invalid.")
            return {name: np.asarray(archive[name]) for name in archive.files}

    def _read_metadata(
        self,
        marker: Mapping[str, Any],
    ) -> tuple[dict[str, object], ...]:
        raw = _mapping(_load_json(marker["metadata"]), name="chunk metadata")
        _exact_keys(raw, {"records"}, name="chunk metadata")
        return tuple(
            dict(_mapping(value, name="chunk metadata record"))
            for value in _sequence(raw["records"], name="chunk metadata records")
        )


class FinalDatasetAssembler:
    """Validate the complete compact chunk inventory before domain publication."""

    def __init__(
        self,
        *,
        frame_count: int,
        camera_ids: Sequence[str],
        attempt_token: str,
    ) -> None:
        self.frame_count = _positive_integer(frame_count, name="frame_count")
        self.camera_ids = tuple(
            _identifier(value, name="camera_id") for value in camera_ids
        )
        if not self.camera_ids or len(self.camera_ids) != len(set(self.camera_ids)):
            raise ValueError("camera_ids must be non-empty and unique.")
        self.attempt_token = _identifier(attempt_token, name="attempt_token")

    def validate(
        self,
        readers: Iterable[ChunkReader],
    ) -> tuple[ValidatedChunk, ...]:
        """Validate exact coverage through caller-retained, single-scan readers."""
        reader_tuple = tuple(readers)
        if any(not isinstance(reader, ChunkReader) for reader in reader_tuple):
            raise TypeError("Final assembly requires retained ChunkReader values.")
        validated = tuple(
            reader.validate(expected_attempt_token=self.attempt_token)
            for reader in reader_tuple
        )
        if not validated:
            raise ValueError("Final compact dataset requires at least one chunk.")
        actual = [key for chunk in validated for key in chunk.keys]
        expected = [
            RenderSampleKey(frame_index, camera_id)
            for frame_index in range(self.frame_count)
            for camera_id in self.camera_ids
        ]
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            unexpected = sorted(set(actual) - set(expected))
            duplicates = len(actual) - len(set(actual))
            raise ValueError(
                "Compact dataset inventory mismatch; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
                f"duplicates={duplicates}."
            )
        return validated


@dataclass(frozen=True, slots=True)
class DatasetPerformanceBudget:
    """Config-owned measurable resource limits for one dataset stage."""

    maximum_wall_seconds: float
    maximum_published_bytes: int
    maximum_published_fraction_of_dense_reference: float
    maximum_nht_invocations: int
    maximum_background_cache_misses: int
    maximum_complete_array_scans_per_sample: int
    maximum_batch_frames: int
    execution_device: str
    require_cuda: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_wall_seconds, bool)
            or not isinstance(self.maximum_wall_seconds, (int, float))
            or not math.isfinite(float(self.maximum_wall_seconds))
            or self.maximum_wall_seconds <= 0.0
        ):
            raise ValueError("maximum_wall_seconds must be finite and positive.")
        for name in (
            "maximum_published_bytes",
            "maximum_nht_invocations",
            "maximum_background_cache_misses",
            "maximum_complete_array_scans_per_sample",
            "maximum_batch_frames",
        ):
            _positive_integer(getattr(self, name), name=name)
        _identifier(self.execution_device, name="execution_device")
        if (
            isinstance(self.maximum_published_fraction_of_dense_reference, bool)
            or not isinstance(
                self.maximum_published_fraction_of_dense_reference,
                (int, float),
            )
            or not math.isfinite(
                float(self.maximum_published_fraction_of_dense_reference)
            )
            or not 0.0 < self.maximum_published_fraction_of_dense_reference <= 1.0
        ):
            raise ValueError(
                "maximum_published_fraction_of_dense_reference must be in (0,1]."
            )
        if not isinstance(self.require_cuda, bool):
            raise TypeError("require_cuda must be boolean.")


@dataclass(frozen=True, slots=True)
class DatasetPerformanceMetrics:
    """Machine-readable measured resource evidence for one completed stage."""

    domain: str
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    execution_device: str
    cuda_peak_bytes: int
    nht_invocations: int
    background_cache_misses: int
    complete_array_scans: int
    generated_bytes: int
    published_bytes: int
    dense_reference_bytes: int
    frame_count: int
    camera_count: int
    sample_count: int
    schema: str = PERFORMANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERFORMANCE_SCHEMA:
            raise ValueError("Unsupported dataset performance schema.")
        _identifier(self.domain, name="domain")
        _identifier(self.execution_device, name="execution_device")
        for name in ("wall_seconds", "cpu_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "peak_rss_bytes",
            "cuda_peak_bytes",
            "nht_invocations",
            "background_cache_misses",
            "complete_array_scans",
            "generated_bytes",
            "published_bytes",
            "dense_reference_bytes",
            "frame_count",
            "camera_count",
            "sample_count",
        ):
            _nonnegative_integer(getattr(self, name), name=name)

    def validate_budget(self, budget: DatasetPerformanceBudget) -> None:
        """Fail closed when measured evidence exceeds configured authority."""
        if self.wall_seconds > budget.maximum_wall_seconds:
            raise ValueError("Dataset stage exceeded maximum_wall_seconds.")
        if self.published_bytes > budget.maximum_published_bytes:
            raise ValueError("Dataset stage exceeded maximum_published_bytes.")
        if self.nht_invocations > budget.maximum_nht_invocations:
            raise ValueError("Dataset stage exceeded maximum_nht_invocations.")
        if self.background_cache_misses > budget.maximum_background_cache_misses:
            raise ValueError("Dataset stage exceeded maximum_background_cache_misses.")
        if self.sample_count == 0:
            raise ValueError("Completed dataset performance evidence has no samples.")
        if (
            self.complete_array_scans
            > budget.maximum_complete_array_scans_per_sample * self.sample_count
        ):
            raise ValueError(
                "Dataset stage exceeded maximum_complete_array_scans_per_sample."
            )
        if self.dense_reference_bytes <= 0:
            raise ValueError("Dataset performance evidence lacks a dense reference size.")
        if (
            self.published_bytes / self.dense_reference_bytes
            > budget.maximum_published_fraction_of_dense_reference
        ):
            raise ValueError(
                "Dataset stage exceeded its maximum published/dense byte fraction."
            )
        if budget.require_cuda and not self.execution_device.startswith("cuda"):
            raise ValueError("Dataset production budget requires explicit CUDA execution.")
        if self.execution_device != budget.execution_device:
            raise ValueError("Dataset execution device differs from configured authority.")

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON representation."""
        return {
            "schema": self.schema,
            "domain": self.domain,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "execution_device": self.execution_device,
            "cuda_peak_bytes": self.cuda_peak_bytes,
            "nht_invocations": self.nht_invocations,
            "background_cache_misses": self.background_cache_misses,
            "complete_array_scans": self.complete_array_scans,
            "generated_bytes": self.generated_bytes,
            "published_bytes": self.published_bytes,
            "dense_reference_bytes": self.dense_reference_bytes,
            "frame_count": self.frame_count,
            "camera_count": self.camera_count,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> DatasetPerformanceMetrics:
        """Parse only the current strict performance schema."""
        raw = _mapping(value, name="dataset performance metrics")
        keys = {
            "schema",
            "domain",
            "wall_seconds",
            "cpu_seconds",
            "peak_rss_bytes",
            "execution_device",
            "cuda_peak_bytes",
            "nht_invocations",
            "background_cache_misses",
            "complete_array_scans",
            "generated_bytes",
            "published_bytes",
            "dense_reference_bytes",
            "frame_count",
            "camera_count",
            "sample_count",
        }
        _exact_keys(raw, keys, name="dataset performance metrics")
        return cls(
            schema=_exact_string(raw["schema"], name="schema"),
            domain=_identifier(raw["domain"], name="domain"),
            wall_seconds=_nonnegative_float(raw["wall_seconds"], name="wall_seconds"),
            cpu_seconds=_nonnegative_float(raw["cpu_seconds"], name="cpu_seconds"),
            peak_rss_bytes=_nonnegative_integer(raw["peak_rss_bytes"], name="peak_rss_bytes"),
            execution_device=_identifier(
                raw["execution_device"],
                name="execution_device",
            ),
            cuda_peak_bytes=_nonnegative_integer(raw["cuda_peak_bytes"], name="cuda_peak_bytes"),
            nht_invocations=_nonnegative_integer(raw["nht_invocations"], name="nht_invocations"),
            background_cache_misses=_nonnegative_integer(
                raw["background_cache_misses"], name="background_cache_misses"
            ),
            complete_array_scans=_nonnegative_integer(
                raw["complete_array_scans"], name="complete_array_scans"
            ),
            generated_bytes=_nonnegative_integer(raw["generated_bytes"], name="generated_bytes"),
            published_bytes=_nonnegative_integer(raw["published_bytes"], name="published_bytes"),
            dense_reference_bytes=_nonnegative_integer(
                raw["dense_reference_bytes"], name="dense_reference_bytes"
            ),
            frame_count=_nonnegative_integer(raw["frame_count"], name="frame_count"),
            camera_count=_nonnegative_integer(raw["camera_count"], name="camera_count"),
            sample_count=_nonnegative_integer(raw["sample_count"], name="sample_count"),
        )


class PerformanceTimer:
    """Monotonic wall/CPU/RSS measurement scoped to one stage execution."""

    def __init__(self) -> None:
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()

    def elapsed(self) -> tuple[float, float, int]:
        """Return wall seconds, CPU seconds and process peak RSS bytes."""
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss_bytes = rss if os.uname().sysname == "Darwin" else rss * 1024
        return (
            time.perf_counter() - self._wall_start,
            time.process_time() - self._cpu_start,
            peak_rss_bytes,
        )


def write_performance_metrics(
    path: Path,
    *,
    metrics: DatasetPerformanceMetrics,
    budget: DatasetPerformanceBudget,
) -> Path:
    """Validate and atomically publish one stage performance evidence file."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Performance evidence already exists: {path}")
    metrics.validate_budget(budget)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, metrics.to_dict())
    return path


def load_performance_metrics(
    path: Path,
    *,
    budget: DatasetPerformanceBudget,
) -> DatasetPerformanceMetrics:
    """Strictly reload measured evidence and reapply current config authority."""
    metrics = DatasetPerformanceMetrics.from_dict(_load_json(path))
    metrics.validate_budget(budget)
    return metrics


def sparse_delta_from_composite(
    *,
    key: RenderSampleKey,
    background: BackgroundArrays,
    rgb: NDArray[np.generic],
    alpha: NDArray[np.generic],
    depth: NDArray[np.generic],
    instance_ids: NDArray[np.generic],
) -> ForegroundDelta:
    """Extract visible positive-instance pixels from one composed render."""
    height, width = background.height, background.width
    rgb_value = _array(rgb, np.float32, (height, width, 3), name="composite.rgb")
    alpha_value = _array(
        alpha, np.float32, (height, width, 1), name="composite.alpha"
    )
    depth_value = _array(
        depth, np.float32, (height, width, 1), name="composite.depth"
    )
    ids_value = _array(
        instance_ids, np.int32, (height, width), name="composite.instance_ids"
    )
    visible = ids_value.reshape(-1) > 0
    pixels = np.flatnonzero(visible).astype(np.int32, copy=False)
    return ForegroundDelta(
        key=key,
        pixel_indices=pixels,
        rgb=rgb_value.reshape(-1, 3)[visible],
        alpha=alpha_value.reshape(-1)[visible],
        depth=depth_value.reshape(-1)[visible],
        instance_ids=ids_value.reshape(-1)[visible],
    )


def materialize_logical_sample(
    background: BackgroundArrays,
    delta: ForegroundDelta,
) -> LogicalRenderSample:
    """Reconstruct the exact logical full-frame sample on demand."""
    if delta.key.camera_id != background.camera_id:
        raise ValueError("Foreground delta camera disagrees with its background.")
    rgb = np.array(background.rgb, copy=True)
    alpha = np.array(background.alpha, copy=True)
    depth = np.array(background.depth, copy=True)
    instance_ids: NDArray[np.int32] = np.zeros(
        (background.height, background.width),
        dtype=np.int32,
    )
    pixels = delta.pixel_indices.astype(np.int64, copy=False)
    rgb.reshape(-1, 3)[pixels] = delta.rgb
    alpha.reshape(-1)[pixels] = delta.alpha
    depth.reshape(-1)[pixels] = delta.depth
    instance_ids.reshape(-1)[pixels] = delta.instance_ids
    return LogicalRenderSample(
        key=delta.key,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        instance_ids=instance_ids,
    )


def directory_size_bytes(root: Path) -> int:
    """Return exact regular-file bytes beneath a contained dataset directory."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("directory_size_bytes requires an ordinary directory.")
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Dataset directory contains a symbolic link: {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def discard_working_directory(path: Path, *, owner: Path) -> None:
    """Delete an attempt-local working directory only inside its exact owner."""
    resolved = path.resolve(strict=False)
    owner_resolved = owner.resolve(strict=False)
    if resolved == owner_resolved or not resolved.is_relative_to(owner_resolved):
        raise ValueError("Refusing to discard a path outside its stage owner.")
    if path.is_symlink():
        raise ValueError("Refusing to discard a symbolic-link working directory.")
    if path.exists():
        if not path.is_dir():
            raise ValueError("Working path is not a directory.")
        shutil.rmtree(path)


def _concatenate(
    values: Sequence[ForegroundDelta],
    field: str,
    dtype: type[np.generic],
    *,
    trailing: tuple[int, ...] = (),
) -> NDArray[Any]:
    arrays = [np.asarray(getattr(value, field), dtype=dtype) for value in values]
    if not arrays:
        return np.empty((0, *trailing), dtype=dtype)
    if all(len(array) == 0 for array in arrays):
        return np.empty((0, *trailing), dtype=dtype)
    return np.concatenate(arrays, axis=0)


def _array(
    value: NDArray[Any],
    dtype: type[np.generic],
    shape: tuple[int | None, ...],
    *,
    name: str,
) -> NDArray[Any]:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must use dtype {np.dtype(dtype)}, got {array.dtype}.")
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}.")
    return array


def _load_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"JSON file is not an ordinary file: {path}")

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"Non-finite JSON number {value!r} is forbidden in {path}.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}.")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return dict(value)


def _sequence(value: object, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array.")
    return tuple(value)


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, name: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{name} keys differ; missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}."
        )


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{name} must be a portable path component.")
    return value


def _exact_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative.")
    return float(value)


def _relative(root: Path, path: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def _contained_file(root: Path, value: object, *, name: str) -> Path:
    reference = _relative_reference(value, name=name)
    pure = PurePosixPath(reference)
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is not an ordinary file: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{name} escapes the dataset root.")
    return resolved


def _relative_reference(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    if "\\" in value:
        raise ValueError(f"{name} must use a POSIX relative path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} must be a contained relative path.")
    return value


__all__ = [
    "BACKGROUND_STORE_SCHEMA",
    "FOREGROUND_CHUNK_SCHEMA",
    "PERFORMANCE_SCHEMA",
    "BackgroundArrays",
    "ChunkReader",
    "ChunkWriter",
    "DatasetPerformanceBudget",
    "DatasetPerformanceMetrics",
    "FinalDatasetAssembler",
    "ForegroundDelta",
    "ForegroundDeltaBatch",
    "LogicalRenderSample",
    "PerformanceTimer",
    "RenderSession",
    "RenderSampleKey",
    "SharedBackgroundStore",
    "ValidatedChunk",
    "directory_size_bytes",
    "discard_working_directory",
    "materialize_logical_sample",
    "load_performance_metrics",
    "sparse_delta_from_composite",
    "write_performance_metrics",
]
