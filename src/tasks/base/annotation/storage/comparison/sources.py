"""Freeze every file before opening a task-owned store; never rewrite its source."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ...artifacts import file_reference, read_document, verify_reference
from ...contracts import ArtifactRef, content_digest, relative_path
from ...paths import AnnotationLayout
from ..contracts import Inventory
from ..inventory import inventory, scope_path
from .contracts import CapturedStore, ReadLimits, StoreFile, StoreSpec


class StoreReader(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...
    @property
    def shards(self) -> tuple[Path, ...]: ...
    def __len__(self) -> int: ...
    def key(self, row: int) -> str: ...
    def labels(self, row: int) -> Mapping[str, object]: ...
    def jpeg(self, row: int) -> bytes: ...
    def rgb(self, row: int) -> NDArray[np.uint8]: ...


ReaderFactory = Callable[[StoreSpec, Path], StoreReader]


def scan(layout: AnnotationLayout, spec: StoreSpec, limits: ReadLimits) -> Inventory:
    observed = inventory(layout, (spec.root,), max_entries=limits.maximum_entries)
    if any(entry.kind not in {"file", "directory"} for entry in observed.entries):
        raise ValueError("comparison sources cannot contain symlinks or special files")
    if (
        sum(e.logical_bytes for e in observed.entries if e.kind == "file")
        > limits.maximum_source_bytes
    ):
        raise ValueError("comparison source exceeds its byte limit")
    return observed


def _files(
    layout: AnnotationLayout, spec: StoreSpec, observed: Inventory, shards: set[Path]
) -> tuple[StoreFile, ...]:
    root = scope_path(layout, spec.root)
    paths = {Path(entry.path) for entry in observed.entries if entry.kind == "file"}
    if not shards or not shards <= paths:
        raise ValueError("reader shards must be files inside the captured store")
    return tuple(
        StoreFile(
            relative=path.relative_to(root).as_posix(),
            reference=file_reference(layout, path, role=spec.root.role),
            shared_shard=path in shards,
        )
        for path in sorted(paths)
    )


def capture(
    layout: AnnotationLayout,
    specification: ArtifactRef,
    limits: ReadLimits,
    factory: ReaderFactory,
) -> CapturedStore:
    spec = read_document(layout, specification, StoreSpec)
    before = scan(layout, spec, limits)
    reader = factory(spec, scope_path(layout, spec.root))
    if not 0 < len(reader) <= limits.maximum_frames:
        raise ValueError("comparison frame count exceeds its limit")
    result = CapturedStore(
        specification=specification,
        files=_files(layout, spec, before, set(reader.shards)),
        frame_count=len(reader),
        metadata_digest=content_digest(reader.metadata),
    )
    # Check again after reader construction and hashing, including added files.
    validate_capture(layout, result, limits, factory)
    return result


def validate_capture(
    layout: AnnotationLayout,
    captured: CapturedStore,
    limits: ReadLimits,
    factory: ReaderFactory,
) -> tuple[StoreSpec, StoreReader, Inventory]:
    spec = read_document(layout, captured.specification, StoreSpec)
    observed = scan(layout, spec, limits)
    root = scope_path(layout, spec.root)
    seen: set[Path] = set()
    for item in captured.files:
        relative_path(item.relative)
        path = root / item.relative
        if layout.artifact_path(item.reference) != path or path in seen:
            raise ValueError("captured files must be unique children of their source")
        seen.add(verify_reference(layout, item.reference))
    if seen != {Path(e.path) for e in observed.entries if e.kind == "file"}:
        raise ValueError("comparison source file set changed after capture")
    reader = factory(spec, root)
    if not 0 < len(reader) == captured.frame_count <= limits.maximum_frames:
        raise ValueError("comparison frame count changed or exceeds its limit")
    if content_digest(reader.metadata) != captured.metadata_digest:
        raise ValueError("comparison store metadata changed")
    if set(reader.shards) != {
        root / item.relative for item in captured.files if item.shared_shard
    }:
        raise ValueError("captured shard sharing differs from the native reader")
    return spec, reader, observed


def jpeg_shape(jpeg: bytes, limits: ReadLimits) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(jpeg)) as header:
        if (
            header.format != "JPEG"
            or header.width * header.height > limits.maximum_pixels_per_frame
        ):
            raise ValueError(
                "comparison JPEG exceeds its pixel limit or has another format"
            )
        return header.height, header.width, 3


def rgb_digest(rgb: NDArray[np.uint8], shape: tuple[int, int, int]) -> str:
    if rgb.dtype != np.uint8 or rgb.shape != shape:
        raise ValueError("comparison RGB must retain uint8 HWC shape and channel order")
    return hashlib.sha256(rgb.tobytes()).hexdigest()
