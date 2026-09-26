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
    filter_detections_by_footpoint,
)
from src.tennis_scene.pipeline.components.base import release_inference_memory
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort, SourceVideo
from src.tennis_scene.pipeline.model_assets import PeopleModelConfig
from src.utils.video import OpenCVVideoFrameReader


@dataclass(frozen=True)
class PersonDetectionInput:
    video: SourceVideo
    footpoint_polygon_px: tuple[tuple[float, float], ...] | None


@dataclass(frozen=True)
class PersonDetectionOutput:
    camera_id: str
    frame_offsets: NDArray[np.int64]
    boxes_xyxy: NDArray[np.float32]
    confidence: NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.frame_offsets.ndim != 1 or len(self.frame_offsets) < 2 or self.frame_offsets.dtype != np.int64:
            raise ValueError("Detections require one offset per source frame plus the end")
        if self.frame_offsets[0] != 0 or (np.diff(self.frame_offsets) < 0).any() or self.frame_offsets[-1] != len(self.confidence):
            raise ValueError("Invalid detection offsets")
        if self.boxes_xyxy.shape != (len(self.confidence), 4) or self.boxes_xyxy.dtype != np.float32 or self.confidence.dtype != np.float32:
            raise ValueError("Invalid detection arrays")
        if not np.isfinite(self.boxes_xyxy).all() or not np.isfinite(self.confidence).all() or (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("Detection values must be finite")


class PersonDetectionModule:
    io = ComponentIO("person_detection", PersonDetectionInput, PersonDetectionOutput,
        {"calibration": InputPort("local_court_calibration")}, "person_detections")

    def __init__(self, config: PeopleModelConfig, *, enabled: bool = True) -> None:
        self.config, self.enabled = config, enabled

    def process(self, inputs: PersonDetectionInput) -> PersonDetectionOutput:
        # A camera excluded from court calibration has no court ROI and never
        # enters reconstruction, so it is not detected rather than detected unfiltered.
        if not self.enabled or inputs.footpoint_polygon_px is None:
            return PersonDetectionOutput(inputs.video.camera_id, np.zeros(inputs.video.num_frames + 1, np.int64),
                np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
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
        try:
            for packet in OpenCVVideoFrameReader(inputs.video.path, max_frames=inputs.video.num_frames):
                if config.detector == "dino":
                    detections = detector.predict(PersonDetectionRequest(packet.frame))
                else:
                    prediction = detector.predict(packet.frame, classes=[0], conf=config.runtime.tracking.yolo_confidence,
                        device=config.runtime.device, verbose=False)[0]
                    detections = PersonDetectionResult(prediction.boxes.xyxy.detach().cpu().numpy().astype(np.float32),
                        prediction.boxes.conf.detach().cpu().numpy().astype(np.float32))
                detections = filter_detections_by_footpoint(detections, inputs.footpoint_polygon_px)
                boxes.append(detections.boxes_xyxy)
                scores.append(detections.scores)
                offsets.append(offsets[-1] + len(detections.scores))
        finally:
            if config.detector == "dino":
                detector.unload()
            del detector
            release_inference_memory(config.runtime.device)
        if len(offsets) != inputs.video.num_frames + 1:
            raise ValueError("Detector did not decode the complete source timeline")
        return PersonDetectionOutput(inputs.video.camera_id, np.asarray(offsets, np.int64), np.concatenate(boxes), np.concatenate(scores))
