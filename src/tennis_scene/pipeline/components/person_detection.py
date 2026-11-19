"""Full-frame DINO person detection before tracking and pose estimation."""

from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.submodules.models import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
    dino_batch_item_requirements,
    dino_source_assets,
)
from src.tasks.person_tracking.duplicate_boxes import DuplicateMerge, merge_person_boxes
from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchAddress,
    BatchSource,
    BatchUnit,
    SourceMemory,
)
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
    batch_model = "dino"
    batch_alias = "detector"
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
        output = _DetectionRows(inputs.video, self.merge_duplicates)
        for packet in OpenCVVideoFrameReader(inputs.video.path, max_frames=inputs.video.num_frames):
            output.write(packet.index, detector.predict(PersonDetectionRequest(packet.frame)))
        return output.finish()

    def batch_adapter(self) -> BatchAdapter | None:
        return self if self.enabled else None

    def open_batch(self, clip_id: str, inputs: PersonDetectionInput, models: ModelBorrower,
                   memory: SourceMemory) -> BatchSource:
        video = inputs.video
        config = self.config.runtime.dino_detector
        shape, item_bytes = dino_batch_item_requirements(
            video.height, video.width, short_side=config.short_side, max_long_side=config.max_long_side)
        original_pixels = video.height * video.width
        # A reader can retain one decoded frame and a replacement.
        memory.reserve(original_pixels * 6 + (video.num_frames + 1) * 416)
        output = _DetectionRows(video, self.merge_duplicates)

        def units() -> Generator[BatchUnit, None, None]:
            for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
                if packet.frame.shape != (video.height, video.width, 3) or packet.frame.dtype != np.uint8:
                    raise ValueError("Detector video frame size or dtype changed")
                yield BatchUnit(BatchAddress(clip_id, video.camera_id, packet.index), shape,
                                PersonDetectionRequest(packet.frame), item_bytes)

        def write(unit: BatchUnit, result: PersonDetectionResult) -> None:
            # Both scatter storage and the concatenated output coexist briefly.
            memory.reserve(len(result.scores) * (2 * (5 * 4 + 8) + (192 if self.merge_duplicates else 0)))
            output.write(unit.address.source_frame, result)

        return BatchSource(units(), write, output.finish)

    def infer_batch(self, units: Sequence[BatchUnit], models: ModelBorrower) -> Sequence[Any]:
        detector: DinoPersonDetector = models.borrow(self.batch_alias)
        detections: list[PersonDetectionResult] = detector.predict_batch(
            [unit.payload for unit in units], batch_size=len(units))
        return detections


class _DetectionRows:
    def __init__(self, video: SourceVideo, merge_duplicates: bool) -> None:
        self.video, self.merge_duplicates = video, merge_duplicates
        self.offsets = [0]
        self.boxes: list[NDArray[np.float32]] = []
        self.scores: list[NDArray[np.float32]] = []
        self.source_rows: list[NDArray[np.int64]] = []
        self.merges: list[DuplicateMerge] = []
        self.source_offset = 0

    def write(self, frame: int, detections: PersonDetectionResult) -> None:
        if frame + 1 != len(self.offsets):
            raise ValueError("DINO frame order changed")
        rows: NDArray[np.int64] = np.arange(self.source_offset, self.source_offset + len(detections.scores), dtype=np.int64)
        self.source_offset += len(rows)
        keep, dropped = merge_person_boxes(frame, rows, detections.boxes_xyxy,
                                            detections.scores, enabled=self.merge_duplicates)
        self.source_rows.append(rows[keep])
        self.merges.extend(dropped)
        self.boxes.append(detections.boxes_xyxy[keep])
        self.scores.append(detections.scores[keep])
        self.offsets.append(self.offsets[-1] + len(keep))

    def finish(self) -> PersonDetectionOutput:
        if len(self.offsets) != self.video.num_frames + 1:
            raise ValueError("Detector did not decode the complete source timeline")
        return PersonDetectionOutput(self.video.camera_id, np.asarray(self.offsets, np.int64), np.concatenate(self.boxes),
                                     np.concatenate(self.scores), np.concatenate(self.source_rows), tuple(self.merges))
