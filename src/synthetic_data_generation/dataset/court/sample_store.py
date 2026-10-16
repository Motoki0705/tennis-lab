"""Published Court JPEG storage, independent of the v1/v2/v3 geometry schema.

Only sparse projection labels, camera/trajectory metadata and generation QA
results survive publication. Renderer RGB/alpha/depth remain attempt-local.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.schema import (
    court_schema_from_dataset_schema,
)
from src.utils.data.float32_store import read_float32
from src.utils.data.image_record_store import ImageRecordStore, ImageRecordWriter
from src.utils.io import save_json_atomic

STORAGE_SCHEMA = "court_image_store_v1"
LEGACY_SAMPLE_FILES = frozenset(
    {
        "directory",
        "rgb",
        "rgb_preview",
        "alpha",
        "alpha_preview",
        "depth",
        "depth_coordinate_space",
        "labels",
    }
)
_LABEL_FIELDS = (
    "sample_index",
    "sample_id",
    "trajectory_group_id",
    "trajectory_id",
    "view_id",
    "trajectory_frame_index",
    "split",
    "camera",
    "projection",
    "metadata",
)


def finish_court_store(
    root: Path, writer: ImageRecordWriter, manifest: Mapping[str, Any]
) -> None:
    """Publish the descriptor last; the surrounding stage owns atomic rename."""
    definition = court_schema_from_dataset_schema(manifest["schema"])
    metadata = {
        key: value
        for key, value in manifest.items()
        if key not in {"samples", "storage"}
    }
    descriptor = {
        "schema": STORAGE_SCHEMA,
        "geometry_schema": definition.dataset_schema,
        "scene_id": manifest["scene_id"],
        "status": "completed",
        "jpeg_quality": 95,
        "renderer_rasters_retained": False,
        "visibility_authority": "generation_alpha_depth_gate",
        "storage": writer.finish(metadata),
    }
    save_json_atomic(descriptor, root / "dataset.json")


def open_court_store(root: Path) -> ImageRecordStore:
    descriptor = json.loads((root / "dataset.json").read_text())
    expected = {
        "schema",
        "geometry_schema",
        "scene_id",
        "status",
        "jpeg_quality",
        "renderer_rasters_retained",
        "visibility_authority",
        "storage",
    }
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != expected
        or descriptor["schema"] != STORAGE_SCHEMA
    ):
        raise ValueError("Invalid Court image-store descriptor.")
    if (
        descriptor["status"] != "completed"
        or descriptor["jpeg_quality"] != 95
        or descriptor["renderer_rasters_retained"] is not False
        or descriptor["visibility_authority"] != "generation_alpha_depth_gate"
    ):
        raise ValueError("Court storage/QA contract changed.")
    store = ImageRecordStore(root / "samples", descriptor["storage"])
    if (
        store.metadata["schema"] != descriptor["geometry_schema"]
        or store.metadata["scene_id"] != descriptor["scene_id"]
        or store.metadata["status"] != "completed"
    ):
        raise ValueError("Court descriptor and packed metadata disagree.")
    court_schema_from_dataset_schema(store.metadata["schema"])
    return store


def read_court_manifest(
    root: Path, *, store: ImageRecordStore | None = None
) -> dict[str, Any]:
    """Dispatch the declared storage schema; never infer it from file presence."""
    descriptor = json.loads((root / "dataset.json").read_text())
    if not isinstance(descriptor, dict):
        raise ValueError("Court manifest must be an object.")
    if descriptor.get("schema") != STORAGE_SCHEMA:
        court_schema_from_dataset_schema(descriptor.get("schema"))
        return descriptor
    selected = open_court_store(root) if store is None else store
    result = dict(selected.metadata)
    if "samples" in result or "storage" in result:
        raise ValueError("Court packed metadata must not duplicate sample storage.")
    fields = result["metadata_fields"]

    def ordered_record(record: dict[str, Any]) -> dict[str, Any]:
        # JSON object order is not an authority. The declared metadata_fields
        # sequence owns the order required by the geometric semantic manifest.
        metadata = record["metadata"]
        if set(metadata) != set(fields) or len(fields) != len(set(fields)):
            raise ValueError("Court sparse metadata disagrees with metadata_fields.")
        return {**record, "metadata": {key: metadata[key] for key in fields}}

    result["rejected_samples"] = [
        ordered_record(record) for record in result["rejected_samples"]
    ]
    samples = []
    for row in range(len(selected)):
        record = selected.record(row)
        if LEGACY_SAMPLE_FILES.intersection(record) or "image_index" in record:
            raise ValueError(
                "Court sparse record contains a redundant storage reference."
            )
        samples.append({**ordered_record(record), "image_index": row})
    result["samples"] = samples
    result["storage"] = descriptor
    return result


def read_court_labels(
    root: Path, record: Mapping[str, Any], *, dataset_schema: str
) -> dict[str, Any]:
    if "image_index" not in record:
        path = (root / record["labels"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve(strict=True)):
            raise ValueError("Court labels escape the dataset owner.")
        return cast(dict[str, Any], json.loads(path.read_text()))
    definition = court_schema_from_dataset_schema(dataset_schema)
    fields = (
        (*_LABEL_FIELDS, "target_court") if "target_court" in record else _LABEL_FIELDS
    )
    return {"schema": definition.sample_schema, **{key: record[key] for key in fields}}


def read_court_rgb(
    root: Path, record: Mapping[str, Any], *, store: ImageRecordStore | None = None
) -> NDArray[np.uint8]:
    if "image_index" in record:
        selected = open_court_store(root) if store is None else store
        image = selected.rgb(record["image_index"])
    else:
        path = (root / record["rgb"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve(strict=True)):
            raise ValueError("Court RGB escapes the dataset owner.")
        rgb = read_float32(path)
        if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("Court RGB must be finite float32 in [0,1].")
        image = np.round(rgb * 255.0).astype(np.uint8)
    if image.shape != (record["height"], record["width"], 3):
        raise ValueError("Court RGB and sparse record dimensions disagree.")
    return image
