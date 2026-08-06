"""Shared contracts and post-processing for person trackers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from src.submodules.vendor.gvhmr.utils.net_utils import moving_average_smooth
from src.submodules.vendor.gvhmr.utils.seq_utils import (
    frame_id_to_mask,
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    rearrange_by_mask,
)


@dataclass(frozen=True)
class TrackRequest:
    """Request for person tracking on a video."""

    video_path: str | Path
    num_tracks: int
    interactive: bool


@dataclass(frozen=True)
class TrackResult:
    """Per-person full-length bounding-box tracks in xyxy pixels."""

    tracks: dict[int, torch.Tensor]
    num_frames: int

    @property
    def track_ids(self) -> list[int]:
        return sorted(self.tracks)

    def bbx_xys(self, track_id: int, *, base_enlarge: float) -> torch.Tensor:
        """Convert a track to downstream ``(center_x, center_y, size)`` boxes."""
        from src.submodules.vendor.gvhmr.utils.hmr_cam import get_bbx_xys_from_xyxy

        if type(base_enlarge) is not float:
            raise TypeError("base_enlarge must be a float.")
        if base_enlarge <= 0.0:
            raise ValueError(f"base_enlarge must be positive, got {base_enlarge}")

        xys: torch.Tensor = get_bbx_xys_from_xyxy(
            self.tracks[track_id], base_enlarge=base_enlarge
        ).float()
        return xys


def select_and_complete_tracks(
    track_history: list[list[dict]], request: TrackRequest, num_frames: int
) -> TrackResult:
    """Select tracks, interpolate missing frames, and apply bbox smoothing."""
    id_to_frame_ids, id_to_bbx_xyxys, ids_by_area = sort_tracks(track_history)
    if not ids_by_area:
        raise RuntimeError(f"No person tracks detected in {request.video_path}")

    if request.interactive:
        from src.submodules.vendor.gvhmr.utils.tracker_selection import select_track_ids

        selected_ids = [
            int(track_id)
            for track_id in select_track_ids(
                track_history, str(request.video_path), ids_by_area
            )
            if int(track_id) in id_to_frame_ids
        ]
    else:
        if request.num_tracks <= 0:
            raise ValueError(f"num_tracks must be positive, got {request.num_tracks}")
        if len(ids_by_area) < request.num_tracks:
            raise RuntimeError(
                f"Requested {request.num_tracks} person tracks, but detected only "
                f"{len(ids_by_area)} in {request.video_path}"
            )
        selected_ids = [int(track_id) for track_id in ids_by_area[: request.num_tracks]]

    if not selected_ids:
        raise RuntimeError("No valid tracks selected")
    tracks = {
        track_id: build_track_tensor(
            id_to_frame_ids[track_id], id_to_bbx_xyxys[track_id], num_frames
        )
        for track_id in selected_ids
    }
    return TrackResult(tracks=tracks, num_frames=num_frames)


def sort_tracks(
    track_history: list[list[dict]],
) -> tuple[dict[int, list[int]], dict[int, np.ndarray], list[int]]:
    """Group detections by id and order ids by accumulated bbox area."""
    id_to_frame_ids: dict[int, list[int]] = defaultdict(list)
    id_to_bbx_lists: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame_id, frame in enumerate(track_history):
        for detection in frame:
            track_id = int(detection["id"])
            id_to_frame_ids[track_id].append(frame_id)
            id_to_bbx_lists[track_id].append(detection["bbx_xyxy"])
    id_to_bbx_xyxys = {key: np.array(value) for key, value in id_to_bbx_lists.items()}
    area_sums = {
        key: float(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1).sum())
        for key, boxes in id_to_bbx_xyxys.items()
    }
    ids_by_area = [
        key
        for key, _ in sorted(area_sums.items(), key=lambda item: item[1], reverse=True)
    ]
    return dict(id_to_frame_ids), id_to_bbx_xyxys, ids_by_area


def build_track_tensor(
    frame_ids: list[int], bbx_xyxys: np.ndarray, num_frames: int
) -> torch.Tensor:
    """Scatter observed boxes, interpolate gaps, and smooth over time."""
    frame_ids_t = torch.tensor(frame_ids)
    bbx_xyxys_t = torch.tensor(bbx_xyxys)
    mask = frame_id_to_mask(frame_ids_t, num_frames)
    track = rearrange_by_mask(bbx_xyxys_t, mask)
    missing_frame_ids = get_frame_id_list_from_mask(~mask)
    track = linear_interpolate_frame_ids(track, missing_frame_ids)
    if not (track.sum(1) != 0).all():
        raise RuntimeError("Track interpolation left empty frames")
    track = moving_average_smooth(track, window_size=5, dim=0)
    track = moving_average_smooth(track, window_size=5, dim=0)
    return cast(torch.Tensor, track.float())
