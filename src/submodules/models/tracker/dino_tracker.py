"""DINO person detections associated by the existing Ultralytics BoT-SORT."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from src.submodules.models._base.inference_model import BaseInferenceModel
from src.submodules.models.dino.person_detector import (
    DinoPersonDetector,
    PersonDetectionRequest,
    PersonDetectionResult,
)
from src.submodules.models.tracker.common import (
    TrackRequest,
    TrackResult,
    select_and_complete_tracks,
)
from src.utils.video.reader import OpenCVVideoFrameReader, probe_video_info


class BotSortAssociator:
    """Thin adapter around the same BoT-SORT backend used by YOLO.track()."""

    def __init__(self) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT

        self._tracker = BOTSORT(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.1,
                new_track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
                gmc_method="sparseOptFlow",
                proximity_thresh=0.5,
                appearance_thresh=0.8,
                with_reid=False,
                model="auto",
            )
        )

    def update(
        self, detections: PersonDetectionResult, frame_bgr: np.ndarray
    ) -> list[dict[str, Any]]:
        from ultralytics.engine.results import Boxes

        classes: NDArray[np.float32] = np.zeros(
            (len(detections.scores), 1), dtype=np.float32
        )
        box_data = np.concatenate(
            [
                detections.boxes_xyxy,
                detections.scores[:, None],
                classes,
            ],
            axis=1,
        )
        boxes = Boxes(box_data, orig_shape=frame_bgr.shape[:2])
        tracked = self._tracker.update(boxes, frame_bgr)
        if tracked.size == 0:
            return []
        return [
            {
                "id": int(round(float(row[4]))),
                "bbx_xyxy": np.asarray(row[:4], dtype=np.float32),
            }
            for row in tracked
        ]


class DinoPersonTracker(BaseInferenceModel[TrackRequest, TrackResult]):
    """Detect people with DINO and preserve the existing BoT-SORT contract."""

    def __init__(
        self,
        checkpoint: str | Path,
        repository: str | Path,
        *,
        device: str | torch.device,
        confidence: float,
        short_side: int,
        max_long_side: int,
    ) -> None:
        super().__init__(device)
        self._detector = DinoPersonDetector(
            checkpoint=checkpoint,
            repository=repository,
            device=self.device,
            confidence=confidence,
            short_side=short_side,
            max_long_side=max_long_side,
        )

    def _load_impl(self) -> None:
        self._detector.load()

    def _unload_impl(self) -> None:
        self._detector.unload()

    def _predict_impl(self, request: TrackRequest) -> TrackResult:
        video_path = Path(request.video_path)
        expected_frames = probe_video_info(video_path).frame_count
        associator = BotSortAssociator()
        track_history: list[list[dict[str, Any]]] = []
        frames = OpenCVVideoFrameReader(video_path)
        for packet in tqdm(frames, total=expected_frames, desc="DINO + BoT-SORT"):
            detections = self._detector.predict(
                PersonDetectionRequest(frame_bgr=packet.frame)
            )
            track_history.append(associator.update(detections, packet.frame))
        if len(track_history) != expected_frames:
            raise RuntimeError(
                f"Decoded {len(track_history)} frames from {video_path}, "
                f"but video metadata reports {expected_frames}"
            )
        return select_and_complete_tracks(track_history, request, expected_frames)
