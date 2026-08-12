"""Strict consumer for PR #729 canonical synthetic Court datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import torch
from PIL import Image

from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
    COURT_SAMPLE_SCHEMA,
)
from src.tasks.court_detection.configuration import SyntheticCourtSourceConfig
from src.tasks.court_detection.data.contracts import (
    CourtInputCapability,
    CourtInputSpec,
    CourtInstance2D,
    CourtKeypointChannels,
    CourtRawSample,
    CourtSampleMetadata,
    CourtSampleRecord,
    CourtSourceSplit,
)
from src.tasks.court_detection.data.target_generation.store import (
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
    CourtDerivedTargetStore,
)

_SYNTHETIC_KP_SCHEMA = "synthetic_symmetric_kp7"
_SYNTHETIC_FLIP_PERMUTATION = (1, 0, 3, 2, 5, 4, 6)
_DATASET_KEYS = {
    "schema",
    "status",
    "scene_id",
    "profile",
    "seed",
    "sampling_policy",
    "metadata_fields",
    "trajectory_groups",
    "samples",
    "rejected_samples",
    "metrics",
    "diagnostics",
}
_SAMPLE_RECORD_KEYS = {
    "sample_index",
    "sample_id",
    "trajectory_group_id",
    "trajectory_id",
    "view_id",
    "trajectory_frame_index",
    "split",
    "shard_id",
    "width",
    "height",
    "camera",
    "projection",
    "directory",
    "rgb",
    "rgb_preview",
    "alpha",
    "alpha_preview",
    "depth",
    "depth_coordinate_space",
    "labels",
    "metadata",
}
_LABEL_KEYS = {
    "schema",
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
}


class SyntheticCourtInput:
    """Load accepted canonical samples using manifest-published paths only."""

    def __init__(
        self,
        config: SyntheticCourtSourceConfig,
        *,
        target_store: CourtDerivedTargetStore,
    ) -> None:
        self.config = config
        self.target_store = target_store
        self._spec = CourtInputSpec(
            source_kind="synthetic_court",
            source_schema=COURT_DATASET_SCHEMA,
            capabilities=frozenset(
                {
                    CourtInputCapability.KEYPOINT_CHANNELS,
                    CourtInputCapability.COURT_INSTANCES,
                    CourtInputCapability.SEGMENTATION_REFERENCE,
                    CourtInputCapability.LINE_REFERENCE,
                }
            ),
            keypoint_schema=_SYNTHETIC_KP_SCHEMA,
            keypoint_channel_names=tuple(SEMANTIC_CLASS_NAMES),
            keypoint_flip_permutation=_SYNTHETIC_FLIP_PERMUTATION,
        )
        self._records = self._load_manifests()

    @property
    def spec(self) -> CourtInputSpec:
        return self._spec

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
        values = self._records[split]
        if not values:
            raise ValueError(f"Synthetic Court source contains no accepted {split!r} samples.")
        return values

    def load(self, record: CourtSampleRecord) -> CourtRawSample:
        if record.payload.get("source_schema") != COURT_DATASET_SCHEMA:
            raise ValueError("Synthetic Court record belongs to another source schema.")
        labels = self._load_labels(record)
        projection = labels["projection"]
        instances, channels = self._parse_projection(projection, record=record)
        image = self._load_rgb(record)
        width, height = image.size
        resolution = cast(Mapping[str, object], projection).get("resolution")
        if resolution != [width, height]:
            raise ValueError("Synthetic Court RGB resolution disagrees with labels projection.")
        source_sample_id = cast(str, record.payload["source_sample_id"])
        scene_id = cast(str, record.payload["scene_id"])
        return CourtRawSample(
            sample_id=record.sample_id,
            image=image,
            keypoint_channels=channels,
            court_instances=instances,
            dense_target_refs=record.dense_target_refs,
            metadata=CourtSampleMetadata(
                source_kind="synthetic_court",
                source_schema=COURT_DATASET_SCHEMA,
                source_sample_id=source_sample_id,
                scene_id=scene_id,
                provenance={
                    "dataset_root": str(record.payload["dataset_root"]),
                    "trajectory_group_id": record.payload["trajectory_group_id"],
                    "trajectory_id": record.payload["trajectory_id"],
                    "view_id": record.payload["view_id"],
                    "camera_id": cast(Mapping[str, object], labels["camera"])[
                        "camera_id"
                    ],
                    "rgb": str(record.image_path),
                    "labels": str(record.annotation_path),
                },
            ),
        )

    def _load_manifests(self) -> dict[CourtSourceSplit, tuple[CourtSampleRecord, ...]]:
        grouped: dict[CourtSourceSplit, list[CourtSampleRecord]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        global_ids: set[str] = set()
        for scene_id in self.config.scene_ids:
            root = self.config.workspace_root / scene_id / "datasets" / "court"
            manifest_path = root / "dataset.json"
            manifest = self._read_json(manifest_path, name="Synthetic Court dataset")
            if set(manifest) != _DATASET_KEYS:
                raise ValueError("Synthetic Court dataset.json fields changed.")
            if manifest["schema"] != COURT_DATASET_SCHEMA:
                raise ValueError(
                    f"Synthetic Court requires schema {COURT_DATASET_SCHEMA!r}."
                )
            if manifest["status"] != "completed":
                raise ValueError("Synthetic Court dataset stage must be completed.")
            if manifest["scene_id"] != scene_id:
                raise ValueError("Synthetic Court scene_id disagrees with configuration.")
            samples = manifest["samples"]
            if not isinstance(samples, list) or not samples:
                raise ValueError("Synthetic Court manifest requires accepted samples.")
            for raw_record in samples:
                record = self._manifest_record(
                    raw_record,
                    root=root,
                    scene_id=scene_id,
                )
                if record.sample_id in global_ids:
                    raise ValueError("Synthetic Court stable sample IDs must be unique.")
                global_ids.add(record.sample_id)
                grouped[record.split].append(record)
        return {split: tuple(values) for split, values in grouped.items()}

    def _manifest_record(
        self,
        value: object,
        *,
        root: Path,
        scene_id: str,
    ) -> CourtSampleRecord:
        if not isinstance(value, Mapping) or set(value) != _SAMPLE_RECORD_KEYS:
            raise ValueError("Synthetic Court accepted sample record fields changed.")
        source_sample_id = value["sample_id"]
        if not isinstance(source_sample_id, str) or not source_sample_id:
            raise ValueError("Synthetic Court sample_id must be non-empty.")
        raw_split = value["split"]
        split_map: dict[str, CourtSourceSplit] = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }
        if raw_split not in split_map:
            raise ValueError(f"Unsupported Synthetic Court split: {raw_split!r}.")
        split = split_map[cast(str, raw_split)]
        rgb_path = self._resolve_published_path(root, value["rgb"], name="rgb")
        labels_path = self._resolve_published_path(root, value["labels"], name="labels")
        if not rgb_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError("Synthetic Court manifest-published RGB/labels are missing.")
        stable_id = f"{scene_id}:{source_sample_id}"
        derived_key = f"{scene_id}/{source_sample_id}"
        return CourtSampleRecord(
            sample_id=stable_id,
            split=split,
            image_path=rgb_path,
            annotation_path=labels_path,
            derived_key=derived_key,
            dense_target_refs={
                "seg": self.target_store.path_for(
                    source_kind="synthetic_court",
                    derived_key=derived_key,
                    target_schema=SEGMENTATION_TARGET_SCHEMA,
                ),
                "line": self.target_store.path_for(
                    source_kind="synthetic_court",
                    derived_key=derived_key,
                    target_schema=LINE_TARGET_SCHEMA,
                ),
            },
            payload={
                "source_schema": COURT_DATASET_SCHEMA,
                "source_sample_id": source_sample_id,
                "scene_id": scene_id,
                "dataset_root": root,
                "sample_index": value["sample_index"],
                "trajectory_group_id": value["trajectory_group_id"],
                "trajectory_id": value["trajectory_id"],
                "view_id": value["view_id"],
                "manifest_projection": value["projection"],
                "manifest_camera": value["camera"],
                "manifest_metadata": value["metadata"],
            },
        )

    @staticmethod
    def _resolve_published_path(root: Path, value: object, *, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Synthetic Court {name} path must be a non-empty string.")
        pure = PurePosixPath(value)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"Synthetic Court {name} path must be a safe relative path.")
        target = root.joinpath(*pure.parts)
        if not target.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
            raise ValueError(f"Synthetic Court {name} path escapes dataset root.")
        return target

    def _load_labels(self, record: CourtSampleRecord) -> dict[str, object]:
        labels = self._read_json(record.annotation_path, name="Synthetic Court labels")
        if set(labels) != _LABEL_KEYS or labels["schema"] != COURT_SAMPLE_SCHEMA:
            raise ValueError("Synthetic Court labels.json schema/fields changed.")
        if labels["sample_id"] != record.payload["source_sample_id"]:
            raise ValueError("Synthetic Court labels sample_id disagrees with manifest.")
        for key, payload_key in (
            ("projection", "manifest_projection"),
            ("camera", "manifest_camera"),
            ("metadata", "manifest_metadata"),
        ):
            if labels[key] != record.payload[payload_key]:
                raise ValueError(f"Synthetic Court labels {key} disagrees with manifest.")
        return labels

    @staticmethod
    def _load_rgb(record: CourtSampleRecord) -> Image.Image:
        rgb = np.load(record.image_path, allow_pickle=False)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
            raise ValueError("Synthetic Court RGB must be finite [H,W,3].")
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            if np.any(rgb < 0.0) or np.any(rgb > 1.0):
                raise ValueError("Synthetic Court float RGB must be in [0,1].")
            rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)
        elif rgb.dtype == np.uint8:
            rgb_u8 = rgb
        else:
            raise TypeError("Synthetic Court RGB must use float32/float64 or uint8.")
        return Image.fromarray(rgb_u8, mode="RGB")

    def _parse_projection(
        self,
        value: object,
        *,
        record: CourtSampleRecord,
    ) -> tuple[tuple[CourtInstance2D, ...], CourtKeypointChannels]:
        projection = self._exact_mapping(
            value,
            {
                "camera_id",
                "resolution",
                "coverage_modes",
                "visible_class_names",
                "visible_point_count",
                "courts",
            },
            name="projection",
        )
        courts = projection["courts"]
        if not isinstance(courts, Sequence) or isinstance(courts, (str, bytes)) or not courts:
            raise ValueError("Synthetic Court projection.courts must be non-empty.")
        instances: list[CourtInstance2D] = []
        channel_points: list[list[tuple[float, float]]] = [[] for _ in range(7)]
        channel_visible: list[list[bool]] = [[] for _ in range(7)]
        channel_physical: list[list[int]] = [[] for _ in range(7)]
        court_ids: set[str] = set()
        for court_value in courts:
            court = self._exact_mapping(
                court_value,
                {"court_instance_id", "coverage_mode", "classes"},
                name="projection.court",
            )
            court_id = court["court_instance_id"]
            if not isinstance(court_id, str) or not court_id or court_id in court_ids:
                raise ValueError("Synthetic Court instance IDs must be non-empty and unique.")
            court_ids.add(court_id)
            classes = court["classes"]
            if not isinstance(classes, Sequence) or isinstance(classes, (str, bytes)) or len(classes) != 7:
                raise ValueError("Synthetic Court requires exactly seven semantic classes.")
            instance_points = torch.empty((14, 2), dtype=torch.float32)
            instance_visible = torch.zeros(14, dtype=torch.bool)
            seen_physical: set[int] = set()
            for class_id, class_value in enumerate(classes):
                semantic = self._exact_mapping(
                    class_value,
                    {"class_id", "class_name", "renderer_visible", "points"},
                    name="projection.class",
                )
                if semantic["class_id"] != class_id or semantic["class_name"] != SEMANTIC_CLASS_NAMES[class_id]:
                    raise ValueError("Synthetic Court class IDs/names must be ordered exactly 0..6.")
                points = semantic["points"]
                if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or len(points) != 2:
                    raise ValueError("Synthetic Court semantic classes require exactly two points.")
                expected_indices = PHYSICAL_INDICES_BY_CLASS[class_id]
                for point_index, point_value in enumerate(points):
                    point = self._exact_mapping(
                        point_value,
                        {
                            "physical_index",
                            "uv",
                            "camera_depth_m",
                            "scene_xyz_m",
                            "in_front",
                            "in_frame",
                            "renderer_visible",
                        },
                        name="projection.point",
                    )
                    physical_index = point["physical_index"]
                    if physical_index != expected_indices[point_index] or physical_index in seen_physical:
                        raise ValueError("Synthetic Court physical point identity changed.")
                    seen_physical.add(cast(int, physical_index))
                    uv = self._point_xy(point["uv"])
                    in_frame = self._boolean(point["in_frame"], name="point.in_frame")
                    renderer_visible = self._boolean(
                        point["renderer_visible"], name="point.renderer_visible"
                    )
                    instance_points[cast(int, physical_index)] = torch.tensor(uv)
                    instance_visible[cast(int, physical_index)] = in_frame
                    channel_points[class_id].append(uv)
                    channel_visible[class_id].append(renderer_visible)
                    channel_physical[class_id].append(cast(int, physical_index))
            if seen_physical != set(range(14)):
                raise ValueError("Synthetic Court instance must preserve physical points 0..13.")
            instances.append(
                CourtInstance2D(
                    court_instance_id=court_id,
                    physical_indices=torch.arange(14, dtype=torch.long),
                    points_xy=instance_points,
                    point_visible=instance_visible,
                )
            )
        points_per_channel = len(channel_points[0])
        if points_per_channel == 0 or any(
            len(values) != points_per_channel for values in channel_points
        ):
            raise ValueError("Synthetic Court channels require equal non-empty point capacity.")
        channels = CourtKeypointChannels(
            channel_names=tuple(SEMANTIC_CLASS_NAMES),
            points_xy=torch.tensor(channel_points, dtype=torch.float32),
            point_visible=torch.tensor(channel_visible, dtype=torch.bool),
            physical_indices=torch.tensor(channel_physical, dtype=torch.long),
            horizontal_flip_permutation=_SYNTHETIC_FLIP_PERMUTATION,
        )
        return tuple(instances), channels

    @staticmethod
    def _read_json(path: Path, *, name: str) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON object.")
        return cast(dict[str, object], parsed)

    @staticmethod
    def _exact_mapping(value: object, keys: set[str], *, name: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"Synthetic Court {name} fields changed.")
        return value

    @staticmethod
    def _point_xy(value: object) -> tuple[float, float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(type(item) not in (float, int) for item in value)
        ):
            raise ValueError("Synthetic Court point.uv must contain two numbers.")
        point = (float(value[0]), float(value[1]))
        if not np.isfinite(point).all():
            raise ValueError("Synthetic Court point.uv must be finite.")
        return point

    @staticmethod
    def _boolean(value: object, *, name: str) -> bool:
        if type(value) is not bool:
            raise ValueError(f"Synthetic Court {name} must be boolean.")
        return cast(bool, value)


__all__ = ["SyntheticCourtInput"]
