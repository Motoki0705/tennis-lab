"""Strict, explicitly versioned consumer for canonical synthetic Court datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TypedDict, cast

import numpy as np
import torch
from PIL import Image

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
    COURT_SAMPLE_SCHEMA,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
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
from src.utils.schema.court import COURT_KP_NAMES

_V2_DATASET_SCHEMA = "canonical_court_dataset_v2"
_V2_SAMPLE_SCHEMA = "canonical_court_sample_v2"
_V1_KP_SCHEMA = "synthetic_symmetric_kp7"
_V2_KP_SCHEMA = "synthetic_camera_relative_kp14"
_V1_FLIP_PERMUTATION = (1, 0, 3, 2, 5, 4, 6)
_V2_FLIP_PERMUTATION = (1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13)
_V2_OPPOSITE_PHYSICAL_INDICES = (2, 3, 0, 1, 5, 4, 7, 6, 10, 11, 8, 9, 13, 12)
_V2_CHANNEL_NAMES = COURT_KP_NAMES[:14]
_PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

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
_BASE_SAMPLE_RECORD_KEYS = {
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
_BASE_LABEL_KEYS = {
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
_LABEL_MATCH_FIELDS = (
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
_PUBLISHED_FILE_FIELDS = (
    "rgb",
    "rgb_preview",
    "alpha",
    "alpha_preview",
    "depth",
    "labels",
)


class _ParsedPoint(TypedDict):
    physical_index: int
    uv: tuple[float, float]
    in_front: bool
    in_frame: bool
    renderer_visible: bool


class SyntheticCourtInput:
    """Load only the synthetic schema explicitly selected in typed config."""

    def __init__(
        self,
        config: SyntheticCourtSourceConfig,
        *,
        target_store: CourtDerivedTargetStore,
    ) -> None:
        self.config = config
        self.target_store = target_store
        flip_permutation: tuple[int, ...]
        if config.schema == "v1":
            source_schema = COURT_DATASET_SCHEMA
            keypoint_schema = _V1_KP_SCHEMA
            channel_names = tuple(SEMANTIC_CLASS_NAMES)
            flip_permutation = _V1_FLIP_PERMUTATION
        elif config.schema == "v2":
            source_schema = _V2_DATASET_SCHEMA
            keypoint_schema = _V2_KP_SCHEMA
            channel_names = _V2_CHANNEL_NAMES
            flip_permutation = _V2_FLIP_PERMUTATION
        else:  # pragma: no cover - typed configuration is the authority
            raise ValueError(f"Unsupported Synthetic Court schema: {config.schema!r}.")
        self._spec = CourtInputSpec(
            source_kind="synthetic_court",
            source_schema=source_schema,
            capabilities=frozenset(
                {
                    CourtInputCapability.KEYPOINT_CHANNELS,
                    CourtInputCapability.COURT_INSTANCES,
                    CourtInputCapability.SEGMENTATION_REFERENCE,
                    CourtInputCapability.LINE_REFERENCE,
                }
            ),
            keypoint_schema=keypoint_schema,
            keypoint_channel_names=channel_names,
            keypoint_flip_permutation=flip_permutation,
        )
        self._records = self._load_manifests()

    @property
    def spec(self) -> CourtInputSpec:
        return self._spec

    @property
    def available_splits(self) -> tuple[CourtSourceSplit, ...]:
        return tuple(split for split, records in self._records.items() if records)

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
        values = self._records[split]
        if not values:
            raise ValueError(
                f"Synthetic Court source contains no accepted {split!r} samples."
            )
        return values

    def load(self, record: CourtSampleRecord) -> CourtRawSample:
        if record.payload.get("source_schema") != self.spec.source_schema:
            raise ValueError("Synthetic Court record belongs to another source schema.")
        labels = self._load_labels(record)
        projection = labels["projection"]
        instances, channels = self._parse_projection(projection, record=record)
        image = self._load_rgb(record)
        width, height = image.size
        if (width, height) != (
            cast(int, record.payload["width"]),
            cast(int, record.payload["height"]),
        ):
            raise ValueError("Synthetic Court RGB resolution disagrees with manifest.")
        resolution = cast(Mapping[str, object], projection)["resolution"]
        if resolution != [width, height]:
            raise ValueError(
                "Synthetic Court RGB resolution disagrees with labels projection."
            )
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
                source_schema=self.spec.source_schema,
                source_sample_id=source_sample_id,
                scene_id=scene_id,
                provenance={
                    "dataset_root": str(record.payload["dataset_root"]),
                    "dataset_manifest_sha256": record.payload[
                        "dataset_manifest_sha256"
                    ],
                    "source_target_sha256": record.payload[
                        "source_target_sha256"
                    ],
                    "trajectory_group_id": record.payload["trajectory_group_id"],
                    "trajectory_id": record.payload["trajectory_id"],
                    "view_id": record.payload["view_id"],
                    "camera_id": cast(Mapping[str, object], labels["camera"])[
                        "camera_id"
                    ],
                    "rgb": str(record.image_path),
                    "labels": str(record.annotation_path),
                    **(
                        {"target_court": labels["target_court"]}
                        if self.config.schema == "v2"
                        else {}
                    ),
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
        group_splits: dict[tuple[str, str], CourtSourceSplit] = {}
        workspace_root = self.config.workspace_root.resolve(strict=False)
        for scene_id in self.config.scene_ids:
            root = (workspace_root / scene_id / "datasets" / "court").resolve(
                strict=False
            )
            if root == workspace_root or not root.is_relative_to(workspace_root):
                raise ValueError(
                    "Synthetic Court manifest root must remain below workspace_root."
                )
            manifest_path = root / "dataset.json"
            if manifest_path.is_symlink():
                raise ValueError(
                    "Synthetic Court dataset.json must be an ordinary file."
                )
            manifest = self._read_json(manifest_path, name="Synthetic Court dataset")
            if set(manifest) != _DATASET_KEYS:
                raise ValueError("Synthetic Court dataset.json fields changed.")
            if manifest["schema"] != self.spec.source_schema:
                raise ValueError(
                    "Synthetic Court selected schema disagrees with dataset.json; "
                    f"selected={self.config.schema!r}, observed={manifest['schema']!r}."
                )
            if manifest["status"] != "completed":
                raise ValueError("Synthetic Court dataset stage must be completed.")
            if manifest["scene_id"] != scene_id:
                raise ValueError(
                    "Synthetic Court scene_id disagrees with configuration."
                )
            if self.config.schema == "v2":
                self._validate_v2_manifest_envelope(manifest)
            samples = manifest["samples"]
            if not isinstance(samples, list) or not samples:
                raise ValueError("Synthetic Court manifest requires accepted samples.")
            manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            for raw_record in samples:
                record = self._manifest_record(
                    raw_record,
                    root=root,
                    scene_id=scene_id,
                    manifest_digest=manifest_digest,
                )
                if record.sample_id in global_ids:
                    raise ValueError("Synthetic Court stable sample IDs must be unique.")
                global_ids.add(record.sample_id)
                if self.config.schema == "v2":
                    group_id = cast(str, record.payload["trajectory_group_id"])
                    group_key = (scene_id, group_id)
                    previous = group_splits.setdefault(group_key, record.split)
                    if previous != record.split:
                        raise ValueError(
                            "Synthetic Court trajectory group split leakage detected: "
                            f"scene={scene_id!r}, group={group_id!r}."
                        )
                grouped[record.split].append(record)
        if self.config.schema == "v2":
            empty = [split for split, records in grouped.items() if not records]
            if empty:
                raise ValueError(
                    "Synthetic Court v2 requires non-empty train/validation/test "
                    f"splits; empty={empty}."
                )
        return {split: tuple(values) for split, values in grouped.items()}

    @classmethod
    def _validate_v2_manifest_envelope(cls, manifest: Mapping[str, object]) -> None:
        cls._identifier(manifest["profile"], name="profile")
        cls._nonnegative_integer(manifest["seed"], name="seed")
        if not isinstance(manifest["sampling_policy"], Mapping):
            raise TypeError("Synthetic Court sampling_policy must be a mapping.")
        metadata_fields = manifest["metadata_fields"]
        if (
            not isinstance(metadata_fields, list)
            or any(not isinstance(field, str) for field in metadata_fields)
            or len(metadata_fields) != len(set(metadata_fields))
        ):
            raise ValueError(
                "Synthetic Court metadata_fields must be a unique string list."
            )
        if not isinstance(manifest["trajectory_groups"], list):
            raise TypeError("Synthetic Court trajectory_groups must be a list.")
        if not isinstance(manifest["rejected_samples"], list):
            raise TypeError("Synthetic Court rejected_samples must be a list.")
        if not isinstance(manifest["metrics"], Mapping):
            raise TypeError("Synthetic Court metrics must be a mapping.")
        diagnostics = manifest["diagnostics"]
        if not isinstance(diagnostics, list) or any(
            not isinstance(path, str) for path in diagnostics
        ):
            raise TypeError("Synthetic Court diagnostics must be a string list.")
        try:
            json.dumps(manifest, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Synthetic Court v2 manifest must contain finite JSON values."
            ) from error

    def _manifest_record(
        self,
        value: object,
        *,
        root: Path,
        scene_id: str,
        manifest_digest: str,
    ) -> CourtSampleRecord:
        expected_keys = set(_BASE_SAMPLE_RECORD_KEYS)
        if self.config.schema == "v2":
            expected_keys.add("target_court")
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError("Synthetic Court accepted sample record fields changed.")

        source_sample_id = self._identifier(value["sample_id"], name="sample_id")
        trajectory_group_id = self._identifier(
            value["trajectory_group_id"], name="trajectory_group_id"
        )
        trajectory_id = self._identifier(
            value["trajectory_id"], name="trajectory_id"
        )
        view_id = self._identifier(value["view_id"], name="view_id")
        self._nonnegative_integer(value["sample_index"], name="sample_index")
        self._nonnegative_integer(
            value["trajectory_frame_index"], name="trajectory_frame_index"
        )
        width = self._positive_integer(value["width"], name="width", minimum=2)
        height = self._positive_integer(value["height"], name="height", minimum=2)
        raw_split = value["split"]
        split_map: dict[str, CourtSourceSplit] = {
            "train": "train",
            "validation": "val",
            "test": "test",
        }
        if raw_split not in split_map:
            raise ValueError(f"Unsupported Synthetic Court split: {raw_split!r}.")
        split = split_map[cast(str, raw_split)]

        if not isinstance(value["camera"], Mapping):
            raise TypeError("Synthetic Court camera must be a mapping.")
        if not isinstance(value["projection"], Mapping):
            raise TypeError("Synthetic Court projection must be a mapping.")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("Synthetic Court metadata must be a mapping.")
        if self.config.schema == "v2":
            camera = SceneCamera.from_dict(value["camera"])
            if (
                camera.camera_id != source_sample_id
                or camera.source_frame_index != value["sample_index"]
                or camera.width != width
                or camera.height != height
            ):
                raise ValueError(
                    "Synthetic Court v2 camera disagrees with sample identity/resolution."
                )
        published_directory = self._resolve_published_directory(
            root, value["directory"]
        )

        paths: dict[str, Path] = {}
        for field in _PUBLISHED_FILE_FIELDS:
            path = self._resolve_published_path(root, value[field], name=field)
            if self.config.schema == "v2" and not path.is_file():
                raise FileNotFoundError(
                    f"Synthetic Court manifest-published {field} is missing: {path}"
                )
            paths[field] = path
            if self.config.schema == "v2" and not path.resolve(
                strict=False
            ).is_relative_to(published_directory.resolve(strict=True)):
                raise ValueError(
                    f"Synthetic Court v2 {field} path must stay below its "
                    "published sample directory."
                )
        if not paths["rgb"].is_file() or not paths["labels"].is_file():
            raise FileNotFoundError(
                "Synthetic Court manifest-published RGB/labels are missing."
            )
        if self.config.schema == "v2" and value["depth_coordinate_space"] != (
            "metric_scene_metres"
        ):
            raise ValueError(
                "Synthetic Court v2 depth_coordinate_space must be "
                "'metric_scene_metres'."
            )

        target_court_id: str | None = None
        if self.config.schema == "v2":
            target_court_id = self._parse_target_court(value["target_court"])
        digest_payload = {
            "source_schema": self.spec.source_schema,
            "source_sample_id": source_sample_id,
            "width": width,
            "height": height,
            "projection": value["projection"],
            **(
                {"target_court": value["target_court"]}
                if self.config.schema == "v2"
                else {}
            ),
        }
        try:
            digest_bytes = json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Synthetic Court geometry provenance must be finite JSON."
            ) from error
        source_target_digest = hashlib.sha256(digest_bytes).hexdigest()

        stable_id = f"{scene_id}:{source_sample_id}"
        derived_key = f"{scene_id}/{source_sample_id}"
        return CourtSampleRecord(
            sample_id=stable_id,
            split=split,
            image_path=paths["rgb"],
            annotation_path=paths["labels"],
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
                "source_schema": self.spec.source_schema,
                "source_sample_id": source_sample_id,
                "scene_id": scene_id,
                "dataset_root": root.resolve(strict=True),
                "dataset_manifest_sha256": manifest_digest,
                "source_target_sha256": source_target_digest,
                "width": width,
                "height": height,
                "sample_index": value["sample_index"],
                "trajectory_group_id": trajectory_group_id,
                "trajectory_id": trajectory_id,
                "view_id": view_id,
                "manifest_record": dict(value),
                **(
                    {"target_court_id": target_court_id}
                    if target_court_id is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _resolve_published_directory(root: Path, value: object) -> Path:
        target = SyntheticCourtInput._resolve_relative(root, value, name="directory")
        if not target.is_dir() or target.is_symlink():
            raise ValueError(
                "Synthetic Court directory must be a contained ordinary directory."
            )
        return target

    @staticmethod
    def _resolve_published_path(root: Path, value: object, *, name: str) -> Path:
        target = SyntheticCourtInput._resolve_relative(root, value, name=name)
        if target.exists() and (target.is_symlink() or not target.is_file()):
            raise ValueError(
                f"Synthetic Court {name} must be a contained ordinary file."
            )
        return target

    @staticmethod
    def _resolve_relative(root: Path, value: object, *, name: str) -> Path:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError(
                f"Synthetic Court {name} path must be a non-empty POSIX string."
            )
        pure = PurePosixPath(value)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(
                f"Synthetic Court {name} path must be a safe relative path."
            )
        root_resolved = root.resolve(strict=True)
        candidate = root.joinpath(*pure.parts)
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(f"Synthetic Court {name} path escapes dataset root.")
        return candidate

    def _load_labels(self, record: CourtSampleRecord) -> dict[str, object]:
        labels = self._read_json(record.annotation_path, name="Synthetic Court labels")
        expected_keys = set(_BASE_LABEL_KEYS)
        expected_schema = COURT_SAMPLE_SCHEMA
        if self.config.schema == "v2":
            expected_keys.add("target_court")
            expected_schema = _V2_SAMPLE_SCHEMA
        if set(labels) != expected_keys or labels["schema"] != expected_schema:
            raise ValueError("Synthetic Court labels.json schema/fields changed.")
        manifest_record = cast(Mapping[str, object], record.payload["manifest_record"])
        for field in _LABEL_MATCH_FIELDS:
            if labels[field] != manifest_record[field]:
                raise ValueError(
                    f"Synthetic Court labels {field} disagrees with manifest."
                )
        if self.config.schema == "v2":
            if labels["target_court"] != manifest_record["target_court"]:
                raise ValueError(
                    "Synthetic Court labels target_court disagrees with manifest."
                )
            self._parse_target_court(labels["target_court"])
        return labels

    def _load_rgb(self, record: CourtSampleRecord) -> Image.Image:
        rgb = np.load(record.image_path, allow_pickle=False)
        if self.config.schema == "v2":
            expected = (
                cast(int, record.payload["height"]),
                cast(int, record.payload["width"]),
                3,
            )
            if rgb.dtype != np.float32 or rgb.shape != expected:
                raise ValueError(
                    "Synthetic Court v2 RGB must be float32 [H,W,3] matching manifest."
                )
            if not np.isfinite(rgb).all() or np.any(rgb < 0.0) or np.any(rgb > 1.0):
                raise ValueError(
                    "Synthetic Court v2 RGB must be finite and remain in [0,1]."
                )
            rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)
        else:
            if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
                raise ValueError("Synthetic Court RGB must be finite [H,W,3].")
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                if np.any(rgb < 0.0) or np.any(rgb > 1.0):
                    raise ValueError("Synthetic Court float RGB must be in [0,1].")
                rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)
            elif rgb.dtype == np.uint8:
                rgb_u8 = rgb
            else:
                raise TypeError(
                    "Synthetic Court v1 RGB must use float32/float64 or uint8."
                )
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
        if self.config.schema == "v2":
            return self._parse_projection_v2(projection, record=record)
        return self._parse_projection_v1(projection)

    def _parse_projection_v1(
        self,
        projection: Mapping[str, object],
    ) -> tuple[tuple[CourtInstance2D, ...], CourtKeypointChannels]:
        courts = self._required_sequence(projection["courts"], name="projection.courts")
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
            court_id = self._unique_court_id(court["court_instance_id"], court_ids)
            classes = self._required_sequence(
                court["classes"], name="projection.court.classes"
            )
            if len(classes) != 7:
                raise ValueError("Synthetic Court v1 requires seven semantic classes.")
            instance_points = torch.empty((14, 2), dtype=torch.float32)
            instance_in_front = torch.zeros(14, dtype=torch.bool)
            instance_visible = torch.zeros(14, dtype=torch.bool)
            seen_physical: set[int] = set()
            for class_id, class_value in enumerate(classes):
                semantic = self._exact_mapping(
                    class_value,
                    {"class_id", "class_name", "renderer_visible", "points"},
                    name="projection.class",
                )
                if (
                    type(semantic["class_id"]) is not int
                    or semantic["class_id"] != class_id
                    or semantic["class_name"] != SEMANTIC_CLASS_NAMES[class_id]
                ):
                    raise ValueError(
                        "Synthetic Court v1 class IDs/names must be ordered 0..6."
                    )
                points = self._required_sequence(
                    semantic["points"], name="projection.class.points"
                )
                if len(points) != 2:
                    raise ValueError(
                        "Synthetic Court v1 semantic classes require two points."
                    )
                expected_indices = PHYSICAL_INDICES_BY_CLASS[class_id]
                for point_index, point_value in enumerate(points):
                    point = self._parse_point(point_value)
                    physical_index = point["physical_index"]
                    if (
                        physical_index != expected_indices[point_index]
                        or physical_index in seen_physical
                    ):
                        raise ValueError("Synthetic Court physical point identity changed.")
                    seen_physical.add(physical_index)
                    uv = point["uv"]
                    instance_points[physical_index] = torch.tensor(uv)
                    instance_in_front[physical_index] = point["in_front"]
                    instance_visible[physical_index] = (
                        point["in_front"] and point["in_frame"]
                    )
                    channel_points[class_id].append(uv)
                    channel_visible[class_id].append(point["renderer_visible"])
                    channel_physical[class_id].append(physical_index)
            if seen_physical != set(range(14)):
                raise ValueError(
                    "Synthetic Court instance must preserve physical points 0..13."
                )
            instances.append(
                CourtInstance2D(
                    court_instance_id=court_id,
                    physical_indices=torch.arange(14, dtype=torch.long),
                    points_xy=instance_points,
                    point_in_front=instance_in_front,
                    point_visible=instance_visible,
                )
            )
        channels = self._channels(
            names=tuple(SEMANTIC_CLASS_NAMES),
            points=channel_points,
            visible=channel_visible,
            physical=channel_physical,
            flip=_V1_FLIP_PERMUTATION,
        )
        return tuple(instances), channels

    def _parse_projection_v2(
        self,
        projection: Mapping[str, object],
        *,
        record: CourtSampleRecord,
    ) -> tuple[tuple[CourtInstance2D, ...], CourtKeypointChannels]:
        expected_resolution = [
            cast(int, record.payload["width"]),
            cast(int, record.payload["height"]),
        ]
        if projection["resolution"] != expected_resolution:
            raise ValueError(
                "Synthetic Court v2 projection resolution disagrees with manifest."
            )
        if projection["camera_id"] != record.payload["source_sample_id"]:
            raise ValueError(
                "Synthetic Court v2 projection camera_id disagrees with sample_id."
            )
        if (
            not isinstance(projection["visible_class_names"], list)
            or any(
                not isinstance(name, str)
                for name in cast(list[object], projection["visible_class_names"])
            )
            or isinstance(projection["visible_point_count"], bool)
            or not isinstance(projection["visible_point_count"], int)
        ):
            raise ValueError(
                "Synthetic Court v2 visible class/point inventories are invalid."
            )
        courts = self._required_sequence(projection["courts"], name="projection.courts")
        instances: list[CourtInstance2D] = []
        channel_points: list[list[tuple[float, float]]] = [[] for _ in range(14)]
        channel_visible: list[list[bool]] = [[] for _ in range(14)]
        channel_physical: list[list[int]] = [[] for _ in range(14)]
        court_ids: set[str] = set()
        coverage_modes: list[object] = []
        renderer_visible_names: set[str] = set()
        renderer_visible_count = 0
        for court_value in courts:
            court = self._exact_mapping(
                court_value,
                {"court_instance_id", "coverage_mode", "classes"},
                name="projection.court",
            )
            court_id = self._unique_court_id(court["court_instance_id"], court_ids)
            if not isinstance(court["coverage_mode"], str):
                raise ValueError("Synthetic Court v2 coverage_mode must be a string.")
            coverage_modes.append(court["coverage_mode"])
            classes = self._required_sequence(
                court["classes"], name="projection.court.classes"
            )
            if len(classes) != 14:
                raise ValueError(
                    "Synthetic Court v2 requires fourteen singleton semantic classes."
                )
            instance_points = torch.empty((14, 2), dtype=torch.float32)
            instance_in_front = torch.zeros(14, dtype=torch.bool)
            instance_visible = torch.zeros(14, dtype=torch.bool)
            semantic_physical: list[int] = []
            for class_id, class_value in enumerate(classes):
                semantic = self._exact_mapping(
                    class_value,
                    {"class_id", "class_name", "renderer_visible", "points"},
                    name="projection.class",
                )
                class_name = _V2_CHANNEL_NAMES[class_id]
                if (
                    type(semantic["class_id"]) is not int
                    or semantic["class_id"] != class_id
                    or semantic["class_name"] != class_name
                ):
                    raise ValueError(
                        "Synthetic Court v2 class IDs/names must remain in semantic "
                        "order 0..13."
                    )
                points = self._required_sequence(
                    semantic["points"], name="projection.class.points"
                )
                if len(points) != 1:
                    raise ValueError(
                        "Synthetic Court v2 semantic classes must be singleton."
                    )
                point = self._parse_point(points[0])
                renderer_visible = point["renderer_visible"]
                if self._boolean(
                    semantic["renderer_visible"], name="class.renderer_visible"
                ) != renderer_visible:
                    raise ValueError(
                        "Synthetic Court v2 class renderer visibility disagrees "
                        "with its singleton point."
                    )
                physical_index = point["physical_index"]
                if physical_index in semantic_physical:
                    raise ValueError(
                        "Synthetic Court v2 physical indices must not be duplicated."
                    )
                semantic_physical.append(physical_index)
                uv = point["uv"]
                geometry_visible = point["in_front"] and point["in_frame"]
                supervision_visible = geometry_visible and renderer_visible
                instance_points[physical_index] = torch.tensor(uv)
                instance_in_front[physical_index] = point["in_front"]
                instance_visible[physical_index] = geometry_visible
                channel_points[class_id].append(uv)
                channel_visible[class_id].append(supervision_visible)
                channel_physical[class_id].append(physical_index)
                if renderer_visible:
                    renderer_visible_names.add(class_name)
                    renderer_visible_count += 1
            physical_order = tuple(semantic_physical)
            if physical_order not in (
                tuple(range(14)),
                _V2_OPPOSITE_PHYSICAL_INDICES,
            ):
                raise ValueError(
                    "Synthetic Court v2 physical indices must be one camera-relative "
                    "0..13 permutation."
                )
            instances.append(
                CourtInstance2D(
                    court_instance_id=court_id,
                    physical_indices=torch.arange(14, dtype=torch.long),
                    points_xy=instance_points,
                    point_in_front=instance_in_front,
                    point_visible=instance_visible,
                )
            )
        if projection["coverage_modes"] != coverage_modes:
            raise ValueError(
                "Synthetic Court v2 coverage_modes disagree with court order."
            )
        expected_visible_names = [
            name for name in _V2_CHANNEL_NAMES if name in renderer_visible_names
        ]
        if projection["visible_class_names"] != expected_visible_names:
            raise ValueError(
                "Synthetic Court v2 visible_class_names inventory is inconsistent."
            )
        if projection["visible_point_count"] != renderer_visible_count:
            raise ValueError(
                "Synthetic Court v2 visible_point_count inventory is inconsistent."
            )
        target_court_id = record.payload.get("target_court_id")
        if sum(instance.court_instance_id == target_court_id for instance in instances) != 1:
            raise ValueError(
                "Synthetic Court v2 target_court must occur exactly once in projection."
            )
        channels = self._channels(
            names=_V2_CHANNEL_NAMES,
            points=channel_points,
            visible=channel_visible,
            physical=channel_physical,
            flip=_V2_FLIP_PERMUTATION,
        )
        return tuple(instances), channels

    @staticmethod
    def _channels(
        *,
        names: tuple[str, ...],
        points: list[list[tuple[float, float]]],
        visible: list[list[bool]],
        physical: list[list[int]],
        flip: tuple[int, ...],
    ) -> CourtKeypointChannels:
        point_capacity = len(points[0])
        if point_capacity == 0 or any(len(values) != point_capacity for values in points):
            raise ValueError(
                "Synthetic Court channels require equal non-empty point capacity."
            )
        return CourtKeypointChannels(
            channel_names=names,
            points_xy=torch.tensor(points, dtype=torch.float32),
            point_visible=torch.tensor(visible, dtype=torch.bool),
            physical_indices=torch.tensor(physical, dtype=torch.long),
            horizontal_flip_permutation=flip,
        )

    @classmethod
    def _parse_target_court(cls, value: object) -> str:
        target = cls._exact_mapping(
            value,
            {
                "binding",
                "resolution_policy",
                "camera_to_court_center_distance_m",
            },
            name="target_court",
        )
        binding = TargetCourtBinding.from_dict(target["binding"])
        court_id: str = binding.court_instance_id
        cls._nonnegative_integer(binding.selection_seed, name="selection_seed")
        if target["resolution_policy"] not in {
            "trajectory_center_court",
            "nearest_camera",
        }:
            raise ValueError("Synthetic Court v2 target resolution_policy is invalid.")
        distance = target["camera_to_court_center_distance_m"]
        if not cls._is_finite_number(distance) or float(cast("float | int", distance)) < 0:
            raise ValueError(
                "Synthetic Court v2 target distance must be finite and non-negative."
            )
        return court_id

    @classmethod
    def _parse_point(cls, value: object) -> _ParsedPoint:
        point = cls._exact_mapping(
            value,
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
        if (
            isinstance(physical_index, bool)
            or not isinstance(physical_index, int)
            or not 0 <= physical_index < 14
        ):
            raise ValueError("Synthetic Court physical_index must lie in 0..13.")
        uv = cls._point_xy(point["uv"])
        if not cls._is_finite_number(point["camera_depth_m"]):
            raise ValueError("Synthetic Court camera_depth_m must be finite.")
        scene_xyz = point["scene_xyz_m"]
        if (
            not isinstance(scene_xyz, Sequence)
            or isinstance(scene_xyz, (str, bytes))
            or len(scene_xyz) != 3
            or any(not cls._is_finite_number(item) for item in scene_xyz)
        ):
            raise ValueError("Synthetic Court scene_xyz_m must contain three numbers.")
        return {
            "physical_index": physical_index,
            "uv": uv,
            "in_front": cls._boolean(point["in_front"], name="point.in_front"),
            "in_frame": cls._boolean(point["in_frame"], name="point.in_frame"),
            "renderer_visible": cls._boolean(
                point["renderer_visible"], name="point.renderer_visible"
            ),
        }

    @staticmethod
    def _read_json(path: Path, *, name: str) -> dict[str, object]:
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON object.")
        return cast(dict[str, object], parsed)

    @staticmethod
    def _exact_mapping(
        value: object, keys: set[str], *, name: str
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"Synthetic Court {name} fields changed.")
        return value

    @staticmethod
    def _required_sequence(value: object, *, name: str) -> Sequence[object]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ValueError(f"Synthetic Court {name} must be a non-empty sequence.")
        return value

    @classmethod
    def _unique_court_id(cls, value: object, seen: set[str]) -> str:
        court_id = cls._portable_identifier(value, name="court_instance_id")
        if court_id in seen:
            raise ValueError("Synthetic Court instance IDs must be unique.")
        seen.add(court_id)
        return court_id

    @staticmethod
    def _point_xy(value: object) -> tuple[float, float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
            or any(not SyntheticCourtInput._is_finite_number(item) for item in value)
        ):
            raise ValueError("Synthetic Court point.uv must contain two finite numbers.")
        return (float(cast("float | int", value[0])), float(cast("float | int", value[1])))

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and bool(np.isfinite(float(value)))
        )

    @staticmethod
    def _boolean(value: object, *, name: str) -> bool:
        if type(value) is not bool:
            raise ValueError(f"Synthetic Court {name} must be boolean.")
        return value

    @staticmethod
    def _identifier(value: object, *, name: str) -> str:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"Synthetic Court {name} must be a trimmed string.")
        return value

    @staticmethod
    def _portable_identifier(value: object, *, name: str) -> str:
        if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError(f"Synthetic Court {name} must be a portable identifier.")
        return value

    @staticmethod
    def _nonnegative_integer(value: object, *, name: str) -> int:
        return SyntheticCourtInput._positive_integer(value, name=name, minimum=0)

    @staticmethod
    def _positive_integer(value: object, *, name: str, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"Synthetic Court {name} must be an integer >= {minimum}."
            )
        return value


__all__ = ["SyntheticCourtInput"]
