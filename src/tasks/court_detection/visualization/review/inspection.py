"""Source-grounded, read-only inspection of saved Court annotations.

The review never equates image bounds with renderer visibility. Dense targets
are explicitly derived geometry, not independently saved masks or QA evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_labels,
    read_court_manifest,
)
from src.tasks.court_detection.data.contracts import (
    CourtInputSpec,
    CourtRawSample,
    CourtSampleRecord,
)
from src.tasks.court_detection.visualization.review.datasets import (
    DENSE_TARGET_SCHEMAS,
    CourtDatasetCatalog,
    CourtDatasetEntry,
    GroundTruthMasks,
)
from src.utils.data.image_record_store import ImageRecordStore

SAMPLE_STATES = frozenset(
    {"out_of_frame", "renderer_not_visible", "duplicate_coordinates"}
)


def record_states(record: CourtSampleRecord) -> frozenset[str]:
    """Find inspection candidates from sparse metadata, without RGB or targets."""
    payload = record.payload
    if payload["source_schema"] == "tennis_court_detector_annotations_v1":
        coordinates = cast(tuple[tuple[float, float], ...], payload["keypoints"])
        renderer_invisible = False
    elif payload["source_schema"] == "canonical_court_dataset_v3":
        manifest = cast(Mapping[str, Any], payload["manifest_record"])
        targets = [
            c
            for c in manifest["projection"]["courts"]
            if c["court_instance_id"] == payload["target_court_id"]
        ]
        if len(targets) != 1:
            raise ValueError("Review must resolve exactly one stored target court.")
        points = [c["points"][0] for c in targets[0]["classes"]]
        coordinates = tuple((float(p["uv"][0]), float(p["uv"][1])) for p in points)
        renderer_invisible = any(
            p["in_front"] and p["in_frame"] and not p["renderer_visible"]
            for p in points
        )
    else:
        raise ValueError("Unsupported annotation schema for Court review filters.")
    states: set[str] = set()
    width, height = cast(int, payload["width"]), cast(int, payload["height"])
    if any(not (0 <= x < width and 0 <= y < height) for x, y in coordinates):
        states.add("out_of_frame")
    if len(set(coordinates)) < len(coordinates):
        states.add("duplicate_coordinates")
    if renderer_invisible:
        states.add("renderer_not_visible")
    return frozenset(states)


def dataset_map(catalog: CourtDatasetCatalog) -> list[dict[str, object]]:
    """Group live catalog splits by source without decoding dataset images."""
    families: dict[str, dict[str, Any]] = {}
    for entry in catalog.entries():
        key = str(entry.path)
        if key not in families:
            source = entry.source_kind
            descriptor = (
                json.loads((entry.path / "dataset.json").read_text())
                if entry.available
                else {}
            )
            families[key] = {
                "label": "TennisCourtDetector"
                if source == "tennis_court_detector"
                else f"Synthetic {entry.scene_id}",
                "source": source,
                "path": key,
                "schema": descriptor.get(
                    "geometry_schema", descriptor.get("schema", entry.published_schema)
                ),
                "storage_schema": descriptor.get("schema"),
                "storage": descriptor.get("storage", {}).get("format", "sparse files"),
                "splits": {},
                "stored_count": descriptor.get("storage", {}).get("count"),
                "excluded_ids": [],
                "trajectory_groups": None,
                "rejected_proposals": None,
                "reason": entry.reason,
            }
            if entry.available and source == "tennis_court_detector":
                # Resolve exclusions from the same typed preset as training.
                from src.tasks.court_detection.data.inputs.tennis_court_detector import (
                    TennisCourtDetectorInput,
                )

                layer = cast(TennisCourtDetectorInput, catalog.input_for(entry))
                families[key]["excluded_ids"] = list(layer.config.excluded_sample_ids)
            if entry.available and source == "synthetic_court":
                manifest = read_court_manifest(entry.path)
                families[key]["stored_count"] = len(manifest["samples"])
                families[key]["trajectory_groups"] = len(manifest["trajectory_groups"])
                families[key]["rejected_proposals"] = len(manifest["rejected_samples"])
        families[key]["splits"][entry.split] = {
            "count": entry.count,
            "available": entry.available,
            "reason": entry.reason,
        }
    return list(families.values())


def saved_annotation(catalog: CourtDatasetCatalog, scene: str) -> dict[str, object]:
    """Return the real stored sparse record, addressed only by catalog scene ID."""
    entry, record = catalog.sample(scene)
    # Verify the canonical source contract before exposing the stored record.
    raw = catalog.input_for(entry).load(record)
    descriptor = json.loads((entry.path / "dataset.json").read_text())
    stored_record = None
    if entry.source_kind == "tennis_court_detector":
        store = ImageRecordStore(entry.path, descriptor["storage"])
        annotation = store.record(cast(int, record.payload["image_index"]))
    else:
        annotation = read_court_labels(
            entry.path,
            cast(Mapping[str, object], record.payload["manifest_record"]),
            dataset_schema=raw.metadata.source_schema,
        )
        if descriptor["schema"] == "court_image_store_v1":
            stored_record = open_court_store(entry.path).record(
                cast(int, record.payload["manifest_record"]["image_index"])
            )
    return {
        "dataset": entry.id,
        "sample": record.sample_id,
        "source_schema": raw.metadata.source_schema,
        "storage_schema": descriptor["schema"],
        "annotation_representation": "logical_labels_from_sparse_record"
        if stored_record is not None
        else "saved_record",
        "annotation_path": str(record.annotation_path),
        "image_path": str(record.image_path),
        "annotation": annotation,
        **({"stored_record": stored_record} if stored_record is not None else {}),
        "provenance": raw.metadata.to_dict(),
        "derived_target_schemas": dict(DENSE_TARGET_SCHEMAS),
    }


def sample_inspection(
    entry: CourtDatasetEntry,
    record: CourtSampleRecord,
    raw: CourtRawSample,
    spec: CourtInputSpec,
    masks: GroundTruthMasks,
) -> dict[str, object]:
    """Keep source visibility, image bounds and KP supervision independent."""
    rows: list[dict[str, Any]] = []
    channels = raw.keypoint_channels
    stored_points: list[Mapping[str, Any]] = []
    synthetic = entry.source_kind == "synthetic_court"
    if synthetic:
        manifest_record = cast(Mapping[str, Any], record.payload["manifest_record"])
        courts = manifest_record["projection"]["courts"]
        targets = [
            c
            for c in courts
            if c["court_instance_id"] == record.payload["target_court_id"]
        ]
        if len(targets) != 1:
            raise ValueError("Review must resolve exactly one stored target court.")
        stored_points = [c["points"][0] for c in targets[0]["classes"]]
    if channels is not None:
        for channel, name in enumerate(channels.channel_names):
            for peak in range(channels.points_xy.shape[1]):
                physical = int(channels.physical_indices[channel, peak])
                if physical < 0:
                    continue
                x, y = channels.points_xy[channel, peak].tolist()
                inside = 0 <= x < raw.image.width and 0 <= y < raw.image.height
                stored = stored_points[channel] if synthetic else None
                front = bool(stored["in_front"]) if stored is not None else None
                renderer = (
                    bool(stored["renderer_visible"]) if stored is not None else None
                )
                visible = bool(channels.point_visible[channel, peak])
                state = (
                    "behind_camera"
                    if front is False
                    else "out_of_frame"
                    if not inside
                    else "renderer_not_visible"
                    if renderer is False
                    else "renderer_visible"
                    if synthetic
                    else "in_frame_visibility_unknown"
                )
                rows.append(
                    {
                        "channel": channel,
                        "name": name,
                        "physical_index": physical,
                        "x": float(x),
                        "y": float(y),
                        "in_frame": inside,
                        "stored_in_frame": stored["in_frame"]
                        if stored is not None
                        else None,
                        "in_front": front,
                        "renderer_visible": renderer,
                        "kp_supervised": visible,
                        "state": state,
                    }
                )
    flags: list[str] = []
    coordinates = [(r["x"], r["y"]) for r in rows]
    if len(set(coordinates)) != len(coordinates):
        flags.append("重複座標あり。点名とraw注釈を確認してください。")
    if any(
        r["stored_in_frame"] is not None and r["in_frame"] != r["stored_in_frame"]
        for r in rows
    ):
        flags.append("保存in_frameと座標から計算した画像境界判定が一致しません。")
    if not rows:
        flags.append("KP注釈がありません。コート不在と判断しないでください。")
    return {
        "dataset": entry.id,
        "source": entry.source_kind,
        "split": entry.split,
        "source_split": record.payload.get(
            "source_split", "validation" if entry.split == "val" else entry.split
        ),
        "sample": record.sample_id,
        "scene": entry.scene_id,
        "source_schema": spec.source_schema,
        "keypoint_schema": spec.keypoint_schema,
        "coordinate_units": "original_image_pixels",
        "annotation_present": channels is not None,
        "stored_visibility": "renderer" if synthetic else "not_provided",
        "pose_teacher": raw.pose_authority is not None,
        "target_court": record.payload.get("target_court_id"),
        "trajectory_group": record.payload.get("trajectory_group_id"),
        "camera_id": raw.metadata.provenance.get("camera_id"),
        "annotation_path": str(record.annotation_path),
        "image_path": str(record.image_path),
        "provenance": raw.metadata.to_dict(),
        "points": rows,
        "counts": dict(Counter(r["state"] for r in rows)),
        "kp_supervised": sum(r["kp_supervised"] for r in rows),
        "flags": flags,
        "targets": [
            {
                "kind": kind,
                "schema": schema,
                "stored": False,
                "available": getattr(masks, kind) is not None,
            }
            for kind, schema in DENSE_TARGET_SCHEMAS.items()
        ],
    }
