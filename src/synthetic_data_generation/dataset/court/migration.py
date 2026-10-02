"""One-time migration of existing renderer outputs through the production writer.

New generation writes JPEG shards directly in assembler.py. This module never
renders and never deletes the source; publication/deletion belongs to the caller.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.synthetic_data_generation.dataset.court.assembler import validate_court_dataset
from src.synthetic_data_generation.dataset.court.sample_store import (
    LEGACY_SAMPLE_FILES,
    finish_court_store,
    read_court_manifest,
    read_court_rgb,
)
from src.synthetic_data_generation.dataset.runtime import directory_size_bytes
from src.synthetic_data_generation.pipeline.locking import scene_write_lock
from src.utils.data.image_record_store import ImageRecordWriter, encode_rgb_jpeg
from src.utils.io import save_json_atomic


def update_publication_bytes(root: Path) -> None:
    """Keep historical generation timings, replacing only current owner bytes."""
    path = root / "diagnostics/performance.json"
    performance = json.loads(path.read_text())
    for _ in range(10):
        size = directory_size_bytes(root)
        performance["metrics"]["published_bytes"] = size
        performance["metrics"]["dense_reference_bytes"] = size
        save_json_atomic(performance, path)
        if directory_size_bytes(root) == size:
            return
    raise RuntimeError("Court publication byte accounting did not converge.")


def migrate_court_store(
    source: Path, destination: Path, *, workers: int = 4
) -> dict[str, object]:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if (
        source == destination
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise ValueError("Court migration requires disjoint source/destination owners.")
    if destination.exists() or not 1 <= workers <= 8:
        raise ValueError("Destination must be new and workers must be in 1..8.")
    with scene_write_lock(source.parents[1]):
        original_bytes = (source / "dataset.json").read_bytes()
        manifest = read_court_manifest(source)
        if "storage" in manifest:
            raise ValueError("Court dataset is already a JPEG store.")
        # This is the final chance to independently verify alpha/depth visibility.
        validate_court_dataset(source)
        before = directory_size_bytes(source)
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source / "diagnostics", destination / "diagnostics")
        samples = manifest["samples"]

        def prepare(record: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
            image = read_court_rgb(source, record)
            sparse = {
                key: value
                for key, value in record.items()
                if key not in LEGACY_SAMPLE_FILES
            }
            return encode_rgb_jpeg(image), sparse

        with (
            ImageRecordWriter(destination / "samples") as writer,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            # Bound in-flight image buffers while preserving authoritative order.
            for start in range(0, len(samples), workers * 2):
                for jpeg, sparse in pool.map(
                    prepare, samples[start : start + workers * 2]
                ):
                    writer.append(jpeg, sparse)
                if start % 200 == 0:
                    print(
                        f"[court-migration] {source.parents[1].name}: {min(start + workers * 2, len(samples))}/{len(samples)}",
                        flush=True,
                    )
            finish_court_store(destination, writer, manifest)
        update_publication_bytes(destination)
        validate_court_dataset(destination)
        restored = read_court_manifest(destination)
        restored.pop("storage")
        expected = dict(manifest)
        expected["samples"] = [
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key not in LEGACY_SAMPLE_FILES
                },
                "image_index": index,
            }
            for index, record in enumerate(samples)
        ]
        if restored != expected:
            raise ValueError(
                "Court migration changed sparse geometry, splits or QA metadata."
            )
        if (source / "dataset.json").read_bytes() != original_bytes:
            raise RuntimeError("Court source changed during migration.")
        return {
            "source": str(source),
            "destination": str(destination),
            "samples": len(samples),
            "original_bytes": before,
            "packed_bytes": directory_size_bytes(destination),
            "source_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "sparse_labels_equal": True,
            "source_raster_visibility_verified": True,
            "new_jpegs_decoded": True,
        }
