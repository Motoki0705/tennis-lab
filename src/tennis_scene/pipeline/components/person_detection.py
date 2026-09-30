"""Replaceable frame detector; no tracking or pose estimation is performed here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)
from src.tasks.person_tracking.duplicate_boxes import DuplicateMerge, merge_person_boxes
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.contracts import ComponentIO, SourceVideo
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


class PersonDetectionModule:
    io = ComponentIO("person_detection", PersonDetectionInput, PersonDetectionOutput,
        {}, "person_detections", version=2)

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True, merge_duplicates: bool = False) -> None:
        self.config, self.enabled = config, enabled
        self.merge_duplicates = merge_duplicates

    def process(self, inputs: PersonDetectionInput) -> PersonDetectionOutput:
        if not self.enabled:
            return PersonDetectionOutput(inputs.video.camera_id, np.zeros(inputs.video.num_frames + 1, np.int64),
                np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.empty(0, np.int64))
        config = self.config
        detector: Any
        if config.detector == "dino":
            detector = DinoPersonDetector(config.dino_checkpoint, config.dino_repository,
                device=config.runtime.device, confidence=config.runtime.dino_detector.confidence,
                short_side=config.runtime.dino_detector.short_side, max_long_side=config.runtime.dino_detector.max_long_side)
        else:
            from ultralytics import YOLO
            detector = YOLO(str(config.yolo_checkpoint))
        offsets = [0]
        boxes: list[NDArray[np.float32]] = []
        scores: list[NDArray[np.float32]] = []
        source_rows: list[NDArray[np.int64]] = []
        merges: list[DuplicateMerge] = []
        source_offset = 0
        try:
            for packet in OpenCVVideoFrameReader(inputs.video.path, max_frames=inputs.video.num_frames):
                if config.detector == "dino":
                    detections = detector.predict(PersonDetectionRequest(packet.frame))
                else:
                    prediction = detector.predict(packet.frame, classes=[0], conf=config.runtime.tracking.yolo_confidence,
                        device=config.runtime.device, verbose=False)[0]
                    detections = PersonDetectionResult(prediction.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                        prediction.boxes.conf.detach().cpu().numpy().astype(np.float32))
                rows: NDArray[np.int64] = np.arange(source_offset, source_offset + len(detections.scores), dtype=np.int64)
                source_offset += len(rows)
                keep, dropped = merge_person_boxes(packet.index, rows, detections.boxes_xyxy,
                                                    detections.scores, enabled=self.merge_duplicates)
                source_rows.append(rows[keep])
                merges.extend(dropped)
                boxes.append(detections.boxes_xyxy[keep])
                scores.append(detections.scores[keep])
                offsets.append(offsets[-1] + len(keep))
        finally:
            if config.detector == "dino":
                detector.unload()
            del detector
            release_inference_memory(config.runtime.device)
        if len(offsets) != inputs.video.num_frames + 1:
            raise ValueError("Detector did not decode the complete source timeline")
        return PersonDetectionOutput(inputs.video.camera_id, np.asarray(offsets, np.int64), np.concatenate(boxes),
                                     np.concatenate(scores), np.concatenate(source_rows), tuple(merges))
