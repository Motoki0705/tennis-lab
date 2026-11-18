"""Full-frame DINO person detection before tracking and pose estimation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    dino_source_assets,
)
from src.tasks.person_tracking.duplicate_boxes import DuplicateMerge, merge_person_boxes
from src.tennis_scene.pipeline.contracts import ComponentIO, SourceVideo
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.video import OpenCVVideoFrameReader


@dataclass(frozen=True)
class PersonDetectionInput:
    video: SourceVideo


@dataclass(frozen=True)
class PersonDetectionOutput:
    camera_id: str
    frame_offsets: NDArray[np.int64]
    boxes_xyxy: NDArray[np.float32]
    confidence: NDArray[np.float32]
    source_rows: NDArray[np.int64] | None = None
    duplicate_merges: tuple[DuplicateMerge, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_offsets.ndim != 1 or len(self.frame_offsets) < 2 or self.frame_offsets.dtype != np.int64:
            raise ValueError("Detections require one offset per source frame plus the end")
        if self.frame_offsets[0] != 0 or (np.diff(self.frame_offsets) < 0).any() or self.frame_offsets[-1] != len(self.confidence):
            raise ValueError("Invalid detection offsets")
        if self.boxes_xyxy.shape != (len(self.confidence), 4) or self.boxes_xyxy.dtype != np.float32 or self.confidence.dtype != np.float32:
            raise ValueError("Invalid detection arrays")
        if not np.isfinite(self.boxes_xyxy).all() or not np.isfinite(self.confidence).all() or (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("Detection values must be finite")
        if self.source_rows is not None and (self.source_rows.shape != self.confidence.shape
                or self.source_rows.dtype != np.int64 or (self.source_rows < 0).any()
                or (np.diff(self.source_rows) <= 0).any()):
            raise ValueError('Detection source rows must be unique and increasing')
        if self.duplicate_merges and self.source_rows is None:
            raise ValueError('Merged detections require original source rows')


class PersonDetectionModule(InferenceComponent[PersonDetectionInput, PersonDetectionOutput]):
    io = ComponentIO("person_detection", PersonDetectionInput, PersonDetectionOutput,
        {}, "person_detections", version=2)

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True, merge_duplicates: bool = False) -> None:
        self.config, self.enabled = config, enabled
        self.merge_duplicates = merge_duplicates

    def cache_identity(self) -> dict[str, bool]:
        return {'enabled': self.enabled, 'merge_duplicates': self.merge_duplicates}

    def model_specs(self) -> dict[str, ModelSpec[DinoPersonDetector]]:
        config = self.config
        if not self.enabled:
            return {}
        extension = os.environ.get("TENNIS_LAB_DINO_EXTENSION")
        native = {} if extension is None else {"extension": Path(extension)}
        if extension is not None and not Path(extension).is_absolute():
            raise ValueError("TENNIS_LAB_DINO_EXTENSION must be an absolute binary path")
        def initialize(model: DinoPersonDetector) -> None:
            if os.environ.get("TENNIS_LAB_DINO_EXTENSION") != extension:
                raise ValueError("DINO extension selection changed since its model declaration")
            model.load()
        return {"detector": ModelSpec(
            "dino", {"checkpoint": config.dino_checkpoint, **dino_source_assets(config.dino_repository), **native},
            "dino_4scale_swin_l", config.runtime.device,
            {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}[config.runtime.dino_detector.precision],
            {"runtime": config.runtime.dino_detector, "extension": extension},
            lambda: DinoPersonDetector(config.dino_checkpoint, config.dino_repository,
                device=config.runtime.device, confidence=config.runtime.dino_detector.confidence,
                short_side=config.runtime.dino_detector.short_side, max_long_side=config.runtime.dino_detector.max_long_side,
                precision=config.runtime.dino_detector.precision),
            initialize=initialize, release=lambda model: model.unload())}

    def process_with_models(self, inputs: PersonDetectionInput, models: ModelBorrower) -> PersonDetectionOutput:
        if not self.enabled:
            return PersonDetectionOutput(inputs.video.camera_id, np.zeros(inputs.video.num_frames + 1, np.int64),
                np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.empty(0, np.int64))
        detector: DinoPersonDetector = models.borrow("detector")
        offsets = [0]
        boxes: list[NDArray[np.float32]] = []
        scores: list[NDArray[np.float32]] = []
        source_rows: list[NDArray[np.int64]] = []
        merges: list[DuplicateMerge] = []
        source_offset = 0
        for packet in OpenCVVideoFrameReader(inputs.video.path, max_frames=inputs.video.num_frames):
            detections = detector.predict(PersonDetectionRequest(packet.frame))
            rows: NDArray[np.int64] = np.arange(source_offset, source_offset + len(detections.scores), dtype=np.int64)
            source_offset += len(rows)
            keep, dropped = merge_person_boxes(packet.index, rows, detections.boxes_xyxy,
                                                detections.scores, enabled=self.merge_duplicates)
            source_rows.append(rows[keep])
            merges.extend(dropped)
            boxes.append(detections.boxes_xyxy[keep])
            scores.append(detections.scores[keep])
            offsets.append(offsets[-1] + len(keep))
        if len(offsets) != inputs.video.num_frames + 1:
            raise ValueError("Detector did not decode the complete source timeline")
        return PersonDetectionOutput(inputs.video.camera_id, np.asarray(offsets, np.int64), np.concatenate(boxes),
                                     np.concatenate(scores), np.concatenate(source_rows), tuple(merges))
