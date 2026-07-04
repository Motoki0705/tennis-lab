"""YOLO-based person tracker (typed port of hmr4d.utils.preproc.tracker)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.submodules.models._base import BaseInferenceModel
from src.submodules.vendor.gvhmr.utils.net_utils import moving_average_smooth
from src.submodules.vendor.gvhmr.utils.seq_utils import (
    frame_id_to_mask,
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    rearrange_by_mask,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.video.reader import probe_video_info

DEFAULT_YOLO_CHECKPOINT = PROJECT_ROOT / "ckpt/yolo/yolov8x.pt"


@dataclass(frozen=True)
class TrackRequest:
    """Request for person tracking on a video.

    Attributes:
        video_path: Source video file.
        num_tracks: Number of tracks to keep (largest accumulated bbox area
            first). Ignored when ``interactive`` is True.
        interactive: Open the track-selection UI (matplotlib) instead of
            picking the top-``num_tracks`` automatically.
    """

    video_path: str | Path
    num_tracks: int = 1
    interactive: bool = False


@dataclass(frozen=True)
class TrackResult:
    """Per-person bounding-box tracks over a full video.

    Attributes:
        tracks: Mapping of track id to interpolated + smoothed boxes
            ``(F, 4)`` in xyxy pixels, covering every frame.
        num_frames: Number of video frames F.
    """

    tracks: dict[int, torch.Tensor] = field(default_factory=dict)
    num_frames: int = 0

    @property
    def track_ids(self) -> list[int]:
        return sorted(self.tracks)

    def bbx_xys(self, track_id: int, base_enlarge: float = 1.2) -> torch.Tensor:
        """Square (center_x, center_y, size) boxes ``(F, 3)`` for one track.

        This is the box parametrization expected by the ViTPose / HMR2 / GVHMR
        models downstream.
        """
        from src.submodules.vendor.gvhmr.utils.hmr_cam import get_bbx_xys_from_xyxy

        xys: torch.Tensor = get_bbx_xys_from_xyxy(
            self.tracks[track_id], base_enlarge=base_enlarge
        ).float()
        return xys


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
        id_to_frame_ids, id_to_bbx_xyxys, ids_by_area = _sort_tracks(track_history)
        if not ids_by_area:
            raise RuntimeError(f"No person tracks detected in {video_path}")

        if request.interactive:
            from src.submodules.vendor.gvhmr.utils.tracker_selection import (
                select_track_ids,
            )

            selected_ids = [
                int(track_id)
                for track_id in select_track_ids(track_history, video_path, ids_by_area)
                if int(track_id) in id_to_frame_ids
            ]
        else:
            selected_ids = [int(track_id) for track_id in ids_by_area[: request.num_tracks]]

        if not selected_ids:
            raise RuntimeError("No valid tracks selected")

        tracks = {
            track_id: _build_track_tensor(
                id_to_frame_ids[track_id], id_to_bbx_xyxys[track_id], num_frames
            )
            for track_id in selected_ids
        }
        return TrackResult(tracks=tracks, num_frames=num_frames)

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


def _sort_tracks(
    track_history: list[list[dict]],
) -> tuple[dict[int, list[int]], dict[int, np.ndarray], list[int]]:
    """Group detections per id and order ids by accumulated bbox area."""
    id_to_frame_ids: dict[int, list[int]] = defaultdict(list)
    id_to_bbx_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_id, frame in enumerate(track_history):
        for det in frame:
            id_to_frame_ids[int(det["id"])].append(frame_id)
            id_to_bbx_lists[int(det["id"])].append(det["bbx_xyxy"])
    id_to_bbx_xyxys = {k: np.array(v) for k, v in id_to_bbx_lists.items()}

    id_area_sum: dict[int, float] = {}
    for k, v in id_to_bbx_xyxys.items():
        bbx_wh = v[:, 2:] - v[:, :2]
        id_area_sum[k] = float((bbx_wh[:, 0] * bbx_wh[:, 1]).sum())
    ids_by_area = [k for k, _ in sorted(id_area_sum.items(), key=lambda kv: kv[1], reverse=True)]

    return dict(id_to_frame_ids), id_to_bbx_xyxys, ids_by_area


def _build_track_tensor(
    frame_ids: list[int],
    bbx_xyxys: np.ndarray,
    num_frames: int,
) -> torch.Tensor:
    """Scatter per-detection boxes to all frames, interpolate gaps, smooth."""
    frame_ids_t = torch.tensor(frame_ids)
    bbx_xyxys_t = torch.tensor(bbx_xyxys)
    mask = frame_id_to_mask(frame_ids_t, num_frames)
    bbx_xyxy_track = rearrange_by_mask(bbx_xyxys_t, mask)
    missing_frame_id_list = get_frame_id_list_from_mask(~mask)
    bbx_xyxy_track = linear_interpolate_frame_ids(bbx_xyxy_track, missing_frame_id_list)
    if not (bbx_xyxy_track.sum(1) != 0).all():
        raise RuntimeError("Track interpolation left empty frames")
    bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
    bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
    return bbx_xyxy_track.float()
