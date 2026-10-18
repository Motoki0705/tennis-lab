"""Sequential equal-input experiments, with correctness checks outside read timers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.data.image_record_store import ImageRecordStore, ImageRecordWriter

from ...artifacts import read_document, write_document
from ...contracts import ArtifactRef, canonical_bytes, content_digest
from ...paths import AnnotationLayout
from ..contracts import InventoryScope
from ..inventory import inventory, scope_path
from .contracts import (
    AttemptTiming,
    CapturedStore,
    ComparisonReport,
    ComparisonRequest,
    PassTiming,
    PolicyResult,
    RowEvidence,
)
from .sources import (
    ReaderFactory,
    StoreReader,
    jpeg_shape,
    rgb_digest,
    validate_capture,
)


class WriteBudget:
    """Conservative file-payload reservation, in addition to the resource supervisor."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.reserved = 0

    def reserve(self, amount: int) -> None:
        if self.reserved + amount > self.maximum:
            raise ValueError("storage comparison write budget exceeded")
        self.reserved += amount


class ConvertedRecords:
    """A comparison-only envelope; it is not a task training dataset format."""

    def __init__(self, root: Path) -> None:
        descriptor = json.loads((root / "descriptor.json").read_bytes())
        self.store = ImageRecordStore(root, descriptor)

    @property
    def metadata(self) -> Mapping[str, object]:
        result: Mapping[str, object] = self.store.metadata
        return result

    @property
    def shards(self) -> tuple[Path, ...]:
        return tuple(self.store.paths)

    def __len__(self) -> int:
        return len(self.store)

    def key(self, row: int) -> str:
        result: str = self.store.record(row)["key"]
        return result

    def labels(self, row: int) -> Mapping[str, object]:
        result: Mapping[str, object] = self.store.record(row)["labels"]
        return result

    def jpeg(self, row: int) -> bytes:
        result: bytes = self.store.jpeg(row)
        return result

    def rgb(self, row: int) -> NDArray[np.uint8]:
        result: NDArray[np.uint8] = self.store.rgb(row)
        return result


def _row(reader: StoreReader, row: int) -> dict[str, object]:
    return {
        "row": row,
        "key": reader.key(row),
        "labels": content_digest(reader.labels(row)),
        "jpeg": hashlib.sha256(reader.jpeg(row)).hexdigest(),
    }


def _convert(
    source: StoreReader,
    root: Path,
    request: ComparisonRequest,
    budget: WriteBudget,
) -> str:
    digest = hashlib.sha256()
    raw_index_bytes = len(canonical_bytes(source.metadata))
    keys: set[str] = set()
    with ImageRecordWriter(root, shard_bytes=request.converted_shard_bytes) as writer:
        for row in range(len(source)):
            key = source.key(row)
            if not key or key in keys:
                raise ValueError("storage comparison requires unique nonempty row keys")
            keys.add(key)
            jpeg = source.jpeg(row)
            jpeg_shape(
                jpeg, request.limits
            )  # Bounds even frames outside the read sample.
            record = {"key": key, "labels": source.labels(row)}
            budget.reserve(len(jpeg))
            # npz's zlib stream, headers and descriptor are bounded conservatively.
            size = len(canonical_bytes(record)) + 128
            raw_index_bytes += size
            budget.reserve(2 * size)
            writer.append(jpeg, record)
            digest.update(canonical_bytes(_row(source, row)) + b"\n")
        budget.reserve(2 * len(canonical_bytes(source.metadata)) + 65536)
        descriptor = writer.finish(source.metadata)
    payload = canonical_bytes(descriptor)
    budget.reserve(len(payload))
    (root / "descriptor.json").write_bytes(payload)
    if (root / "index.npz").stat().st_size > 2 * raw_index_bytes + 65536:
        raise ValueError("converted index exceeded its reserved compression bound")
    return digest.hexdigest()


def _same_rows(reader: StoreReader, expected: str, metadata: str) -> None:
    digest = hashlib.sha256()
    for row in range(len(reader)):
        digest.update(canonical_bytes(_row(reader, row)) + b"\n")
    if digest.hexdigest() != expected or content_digest(reader.metadata) != metadata:
        raise ValueError("storage format changed row identity, JPEG bytes, or teachers")


def _clone(
    template: Path,
    files: tuple[tuple[str, bool], ...],
    target: Path,
    sharing: Literal["copy", "share"],
    budget: WriteBudget,
) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for relative, shard in files:
        source, destination = template / relative, target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if sharing == "share" and shard:
            # No fallback to a copy: unsupported hardlinks are a failed policy.
            os.link(source, destination, follow_symlinks=False)
        else:
            budget.reserve(source.stat().st_size)
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def _read_pass(
    reader: StoreReader,
    rows: tuple[RowEvidence, ...],
    cache: Path | None,
    identity: str,
    decoder: str,
    budget: WriteBudget,
) -> PassTiming:
    elapsed, hits, decoded = 0, 0, 0
    for expected in rows:
        key = content_digest(
            [identity, expected.model_dump(), decoder, "uint8-RGB-HWC"]
        )
        path = None if cache is None else cache / (key + ".npy")
        started = time.perf_counter_ns()
        rgb: NDArray[np.uint8]
        if path is not None and path.exists():
            rgb = np.load(path, allow_pickle=False)
            hits += 1
        else:
            rgb = reader.rgb(expected.row)
            decoded += 1
            if path is not None:
                budget.reserve(rgb.nbytes + 1024)
                with path.open("xb") as stream:
                    np.save(stream, rgb, allow_pickle=False)
        elapsed += time.perf_counter_ns() - started
        # Every hit is validated. Timings do not include checksum work.
        if rgb_digest(rgb, expected.shape) != expected.rgb_digest:
            raise ValueError(
                "storage comparison cache or decoded RGB differs from source"
            )
    return PassTiming(read_ns=elapsed, cache_hits=hits, decoded_frames=decoded)


def compare(
    layout: AnnotationLayout,
    request_ref: ArtifactRef,
    output: Path,
    factory: ReaderFactory,
) -> ComparisonReport:
    request = read_document(layout, request_ref, ComparisonRequest)
    captured = read_document(layout, request.source, CapturedStore)
    spec, source, observed = validate_capture(layout, captured, request.limits, factory)
    source_root = scope_path(layout, spec.root)
    relative = output.relative_to(layout.root).parts
    if (
        len(relative) != 5
        or relative[0] != "campaigns"
        or relative[2] != "jobs"
        or relative[4] != "comparison"
    ):
        raise ValueError("storage comparison output must belong to a campaign job")
    layout.artifact_path(
        ArtifactRef(
            role="annotation",
            path=output.relative_to(layout.root).as_posix(),
            digest="0" * 64,
            size_bytes=0,
        )
    )
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("comparison source and output must not overlap")
    if len(set(request.rows)) != len(request.rows) or max(request.rows) >= len(source):
        raise ValueError("comparison rows must be unique and inside the captured store")
    budget = WriteBudget(request.maximum_written_bytes)
    decoder = f"opencv-{cv2.__version__}/numpy-{np.__version__}/IMREAD_COLOR-BGR2RGB"
    # A failed attempt retains its directory; retry requires a fresh owning job.
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    source_inventory = write_document(layout, evidence / "source.json", observed)
    converted = output / "converted"
    started = time.perf_counter_ns()
    all_rows = _convert(source, converted, request, budget)
    conversion_ns = time.perf_counter_ns() - started
    converted_reader = ConvertedRecords(converted)
    _same_rows(converted_reader, all_rows, captured.metadata_digest)
    rows = tuple(
        RowEvidence(
            row=row,
            key=source.key(row),
            label_digest=content_digest(source.labels(row)),
            jpeg_digest=hashlib.sha256(source.jpeg(row)).hexdigest(),
            rgb_digest=rgb_digest(
                source.rgb(row), jpeg_shape(source.jpeg(row), request.limits)
            ),
            shape=jpeg_shape(source.jpeg(row), request.limits),
        )
        for row in request.rows
    )

    def scope(path: Path) -> InventoryScope:
        return InventoryScope(
            role="annotation", path=path.relative_to(layout.root).as_posix()
        )

    converted_scan = inventory(
        layout, (scope(converted),), max_entries=request.limits.maximum_entries
    )
    converted_inventory = write_document(
        layout, evidence / "converted.json", converted_scan
    )
    converted_files = tuple(
        (
            Path(entry.path).relative_to(converted).as_posix(),
            Path(entry.path) in converted_reader.shards,
        )
        for entry in converted_scan.entries
        if entry.kind == "file"
    )
    results: list[PolicyResult] = []
    formats: tuple[Literal["native", "image_records"], ...] = (
        "native",
        "image_records",
    )
    sharings: tuple[Literal["copy", "share"], ...] = ("copy", "share")
    caches: tuple[Literal["decode", "private_rgb", "shared_rgb"], ...] = (
        "decode",
        "private_rgb",
        "shared_rgb",
    )
    for format_name in formats:
        template = source_root if format_name == "native" else converted
        files = (
            tuple((f.relative, f.shared_shard) for f in captured.files)
            if format_name == "native"
            else converted_files
        )
        for sharing in sharings:
            for cache_policy in caches:
                name = f"{format_name}-{sharing}-{cache_policy}"
                policy_root = output / "policies" / name
                policy_root.mkdir(parents=True)
                timings: list[AttemptTiming] = []
                for attempt in range(request.attempts):
                    owner = policy_root / str(attempt)
                    started = time.perf_counter_ns()
                    _clone(template, files, owner / "store", sharing, budget)
                    clone_ns = time.perf_counter_ns() - started
                    started = time.perf_counter_ns()
                    reader = (
                        factory(spec, owner / "store")
                        if format_name == "native"
                        else ConvertedRecords(owner / "store")
                    )
                    open_ns = time.perf_counter_ns() - started
                    _same_rows(reader, all_rows, captured.metadata_digest)
                    cache = (
                        None
                        if cache_policy == "decode"
                        else (policy_root if cache_policy == "shared_rgb" else owner)
                        / "cache"
                    )
                    if cache is not None:
                        cache.mkdir(exist_ok=True)
                    passes = tuple(
                        _read_pass(
                            reader, rows, cache, captured.sha256, decoder, budget
                        )
                        for _ in range(request.passes)
                    )
                    timings.append(
                        AttemptTiming(clone_ns=clone_ns, open_ns=open_ns, passes=passes)
                    )
                scopes = (
                    spec.root if format_name == "native" else scope(converted),
                    scope(policy_root),
                )
                measured = inventory(
                    layout, scopes, max_entries=request.limits.maximum_entries
                )
                inventory_ref = write_document(
                    layout, evidence / (name + ".json"), measured
                )
                results.append(
                    PolicyResult(
                        format=format_name,
                        shards=sharing,
                        cache=cache_policy,
                        inventory=inventory_ref,
                        attempts=tuple(timings),
                    )
                )
    # Hardlink nlink/ctime changes are expected; content and file sets are not.
    validate_capture(layout, captured, request.limits, factory)
    return ComparisonReport(
        request=request_ref,
        decoder=decoder,
        source_inventory=source_inventory,
        converted_inventory=converted_inventory,
        conversion_ns=conversion_ns,
        all_rows_digest=all_rows,
        rows=rows,
        policies=tuple(results),
        reserved_written_bytes=budget.reserved,
    )
