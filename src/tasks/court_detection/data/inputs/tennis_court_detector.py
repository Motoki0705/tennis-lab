"""Strict adapter for the yastrebksv/TennisCourtDetector data schema."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from PIL import Image

from src.tasks.court_detection.configuration import TennisCourtDetectorSourceConfig
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
    CourtDerivedTargetStore,
    LINE_TARGET_SCHEMA,
    SEGMENTATION_TARGET_SCHEMA,
)

_TCD_KP_SCHEMA = "tennis_court_detector_kp14"
_TCD_CHANNEL_NAMES = tuple(f"physical_{index}" for index in range(14))
_TCD_FLIP_PERMUTATION = (1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13)


class TennisCourtDetectorInput:
    """Convert upstream ordered-14 annotations into canonical raw samples."""

    def __init__(
        self,
        config: TennisCourtDetectorSourceConfig,
        *,
        target_store: CourtDerivedTargetStore,
    ) -> None:
        self.config = config
        self.root = config.root
        self.target_store = target_store
        self._spec = CourtInputSpec(
            source_kind="tennis_court_detector",
            source_schema="tennis_court_detector_annotations_v1",
            capabilities=frozenset(
                {
                    CourtInputCapability.KEYPOINT_CHANNELS,
                    CourtInputCapability.COURT_INSTANCES,
                    CourtInputCapability.SEGMENTATION_REFERENCE,
                    CourtInputCapability.LINE_REFERENCE,
                }
            ),
            keypoint_schema=_TCD_KP_SCHEMA,
            keypoint_channel_names=_TCD_CHANNEL_NAMES,
            keypoint_flip_permutation=_TCD_FLIP_PERMUTATION,
        )
        self._records = self._load_records()

    @property
    def spec(self) -> CourtInputSpec:
        return self._spec

    def records(self, split: CourtSourceSplit) -> tuple[CourtSampleRecord, ...]:
        if split not in self._records:
            raise ValueError(
                f"TennisCourtDetector split {split!r} has no explicit split_mapping."
            )
        return self._records[split]

    def load(self, record: CourtSampleRecord) -> CourtRawSample:
        if record.payload.get("source_schema") != self.spec.source_schema:
            raise ValueError("TennisCourtDetector record belongs to another input schema.")
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB")
        width, height = image.size
        raw_points = record.payload.get("keypoints")
        if not isinstance(raw_points, tuple):
            raise ValueError("TennisCourtDetector record keypoints are unavailable.")
        points = torch.tensor(raw_points, dtype=torch.float32)
        if points.shape != (14, 2) or not bool(torch.isfinite(points).all()):
            raise ValueError("TennisCourtDetector keypoints must be finite [14,2].")
        visible = (
            (points[:, 0] >= 0.0)
            & (points[:, 0] < float(width))
            & (points[:, 1] >= 0.0)
            & (points[:, 1] < float(height))
        )
        physical = torch.arange(14, dtype=torch.long)
        instance = CourtInstance2D(
            court_instance_id=f"{record.sample_id}:court",
            physical_indices=physical,
            points_xy=points,
            point_visible=visible,
        )
        channels = CourtKeypointChannels(
            channel_names=_TCD_CHANNEL_NAMES,
            points_xy=points[:, None, :],
            point_visible=visible[:, None],
            physical_indices=physical[:, None],
            horizontal_flip_permutation=_TCD_FLIP_PERMUTATION,
        )
        return CourtRawSample(
            sample_id=record.sample_id,
            image=image,
            keypoint_channels=channels,
            court_instances=(instance,),
            dense_target_refs=record.dense_target_refs,
            metadata=CourtSampleMetadata(
                source_kind="tennis_court_detector",
                source_schema=self.spec.source_schema,
                source_sample_id=record.sample_id,
                scene_id=None,
                provenance={
                    "annotation": str(record.annotation_path),
                    "image": str(record.image_path),
                    "source_split": record.payload["source_split"],
                },
            ),
        )

    def _load_records(self) -> dict[CourtSourceSplit, tuple[CourtSampleRecord, ...]]:
        records: dict[CourtSourceSplit, tuple[CourtSampleRecord, ...]] = {}
        for split, source_split in self.config.split_mapping.items():
            if source_split is None:
                continue
            records[split] = self._read_source_split(split, source_split)
        return records

    def _read_source_split(
        self, split: CourtSourceSplit, source_split: str
    ) -> tuple[CourtSampleRecord, ...]:
        annotation_path = self.root / f"data_{source_split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"TennisCourtDetector annotation is missing: {annotation_path}"
            )
        parsed = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(
                f"TennisCourtDetector {annotation_path.name} must be a non-empty list."
            )
        result: list[CourtSampleRecord] = []
        seen: set[str] = set()
        for index, value in enumerate(parsed):
            if not isinstance(value, Mapping) or set(value) != {"id", "kps"}:
                raise ValueError(
                    f"TennisCourtDetector record {index} must contain exactly id and kps."
                )
            sample_id = value["id"]
            if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
                raise ValueError("TennisCourtDetector sample IDs must be non-empty and unique.")
            seen.add(sample_id)
            keypoints = self._parse_keypoints(value["kps"], sample_id=sample_id)
            image_path = self._resolve_image(sample_id)
            derived_key = f"{source_split}/{sample_id}"
            result.append(
                CourtSampleRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image_path,
                    annotation_path=annotation_path,
                    derived_key=derived_key,
                    dense_target_refs={
                        "seg": self.target_store.path_for(
                            source_kind="tennis_court_detector",
                            derived_key=derived_key,
                            target_schema=SEGMENTATION_TARGET_SCHEMA,
                        ),
                        "line": self.target_store.path_for(
                            source_kind="tennis_court_detector",
                            derived_key=derived_key,
                            target_schema=LINE_TARGET_SCHEMA,
                        ),
                    },
                    payload={
                        "source_schema": self.spec.source_schema,
                        "source_split": source_split,
                        "keypoints": keypoints,
                    },
                )
            )
        return tuple(result)

    @staticmethod
    def _parse_keypoints(value: object, *, sample_id: str) -> tuple[tuple[float, float], ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 14:
            raise ValueError(f"TennisCourtDetector {sample_id} requires exactly 14 keypoints.")
        points: list[tuple[float, float]] = []
        for point in value:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
                or any(type(item) not in (float, int) for item in point)
            ):
                raise ValueError(f"TennisCourtDetector {sample_id} has an invalid keypoint.")
            points.append((float(point[0]), float(point[1])))
        tensor = torch.tensor(points, dtype=torch.float32)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"TennisCourtDetector {sample_id} keypoints must be finite.")
        return tuple(points)

    def _resolve_image(self, sample_id: str) -> Path:
        images = self.root / "images"
        candidates = [
            images / f"{sample_id}.png",
            images / f"{sample_id}.jpg",
            images / f"{sample_id}.jpeg",
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(
                f"TennisCourtDetector {sample_id} requires exactly one image; found {existing}."
            )
        return existing[0]


__all__ = ["TennisCourtDetectorInput"]
