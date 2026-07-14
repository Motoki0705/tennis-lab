"""YOLO-based person tracker (typed port of hmr4d.utils.preproc.tracker)."""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.submodules.models._base import BaseInferenceModel
from src.submodules.models.tracker.common import (
    TrackRequest,
    TrackResult,
    build_track_tensor,
    select_and_complete_tracks,
    sort_tracks,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.video.reader import probe_video_info

DEFAULT_YOLO_CHECKPOINT = PROJECT_ROOT / "ckpt/yolo/yolov8x.pt"


class YoloPersonTracker(BaseInferenceModel[TrackRequest, TrackResult]):
    """Track persons in a video with YOLO and return per-id xyxy boxes."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_YOLO_CHECKPOINT,
        device: str | torch.device = "auto",
        conf: float = 0.5,
    ) -> None:
        super().__init__(device)
        self.checkpoint = Path(checkpoint)
        self.conf = conf
        self._yolo: object | None = None

    def _load_impl(self) -> None:
        from ultralytics import YOLO

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.checkpoint}")
        self._yolo = YOLO(str(self.checkpoint))

    def _unload_impl(self) -> None:
        self._yolo = None

    def _predict_impl(self, request: TrackRequest) -> TrackResult:
        video_path = str(request.video_path)
        num_frames = probe_video_info(video_path).frame_count

        track_history = self._track(video_path, num_frames)
        return select_and_complete_tracks(track_history, request, num_frames)

    def _track(self, video_path: str, num_frames: int) -> list[list[dict]]:
        """Frame-by-frame YOLO tracking; returns per-frame detection dicts."""
        assert self._yolo is not None
        results = self._yolo.track(  # type: ignore[attr-defined]
            video_path,
            device=self._device,
            conf=self.conf,
            classes=0,  # human
            verbose=False,
            stream=True,
        )
        track_history: list[list[dict]] = []
        for result in tqdm(results, total=num_frames, desc="YOLO tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N,)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                frame = [
                    {"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]}
                    for i in range(len(track_ids))
                ]
            else:
                frame = []
            track_history.append(frame)
        return track_history


# Backward-compatible helper names for callers that imported the old module.
_sort_tracks = sort_tracks
_build_track_tensor = build_track_tensor
