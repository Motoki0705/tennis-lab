"""Shared contracts and post-processing for person trackers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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
    footpoint_polygon_px: tuple[tuple[float, float], ...] | None = None
    stitch_tracklets_in_roi: bool = False
    max_frames: int | None = None


def resolve_track_frame_count(video_frame_count: int, request: TrackRequest) -> int:
    """Resolve the exact number of source frames a tracker must decode."""
    if type(video_frame_count) is not int or video_frame_count <= 0:
        raise ValueError(
            f"video_frame_count must be a positive integer, got {video_frame_count!r}."
        )
    if request.max_frames is None:
        return video_frame_count
    if type(request.max_frames) is not int or request.max_frames <= 0:
        raise ValueError(
            f"TrackRequest.max_frames must be a positive integer or None, "
            f"got {request.max_frames!r}."
        )
    return min(video_frame_count, request.max_frames)


@dataclass(frozen=True)
class TrackResult:
    """Per-person full-length bounding-box tracks in xyxy pixels."""

    tracks: dict[int, torch.Tensor]
    num_frames: int
    observed_masks: dict[int, torch.Tensor] = field(default_factory=dict)

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

    def observed_mask(self, track_id: int) -> torch.Tensor:
        """Return frames backed by detector observations rather than interpolation."""
        if track_id not in self.tracks:
            raise KeyError(f"Unknown track_id {track_id}.")
        if track_id not in self.observed_masks:
            raise RuntimeError(
                f"Track {track_id} has no observation provenance; regenerate tracking."
            )
        mask = self.observed_masks[track_id]
        if mask.dtype != torch.bool or mask.shape != (self.num_frames,):
            raise RuntimeError(
                f"Track {track_id} observation mask must have bool shape "
                f"({self.num_frames},), got {mask.dtype} {tuple(mask.shape)}."
            )
        return mask.clone()


def select_and_complete_tracks(
    track_history: list[list[dict]], request: TrackRequest, num_frames: int
) -> TrackResult:
    """Select tracks, interpolate missing frames, and apply bbox smoothing."""
    if request.stitch_tracklets_in_roi:
        if request.footpoint_polygon_px is None:
            raise ValueError("stitch_tracklets_in_roi requires footpoint_polygon_px.")
        if request.interactive or request.num_tracks != 1:
            raise ValueError(
                "stitch_tracklets_in_roi requires non-interactive single-track "
                "selection."
            )
        track_history = stitch_single_subject_tracklets(track_history)
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
    observed_masks = {
        track_id: frame_id_to_mask(
            torch.tensor(id_to_frame_ids[track_id], dtype=torch.long), num_frames
        ).bool()
        for track_id in selected_ids
    }
    return TrackResult(
        tracks=tracks,
        num_frames=num_frames,
        observed_masks=observed_masks,
    )


def stitch_single_subject_tracklets(
    track_history: list[list[dict]],
) -> list[list[dict]]:
    """Collapse fragmented IDs inside a one-subject ROI into one track.

    The caller must first constrain detections to an ROI containing one target
    subject.  For every observed frame, the largest remaining person box is
    assigned synthetic ID 0.  This preserves actual detector observations
    across BoT-SORT ID changes; missing frames are still explicit and are only
    interpolated later by :func:`build_track_tensor`.
    """
    stitched: list[list[dict]] = []
    for frame in track_history:
        if not frame:
            stitched.append([])
            continue
        candidates: list[tuple[float, dict]] = []
        for detection in frame:
            box = np.asarray(detection["bbx_xyxy"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError(
                    "Tracked person boxes must have finite xyxy shape (4,)."
                )
            width, height = box[2:] - box[:2]
            if width <= 0.0 or height <= 0.0:
                raise ValueError("Tracked person boxes must have positive area.")
            candidates.append((float(width * height), detection))
        _, selected = max(candidates, key=lambda item: item[0])
        stitched.append(
            [
                {
                    **selected,
                    "id": 0,
                    "source_track_id": int(selected["id"]),
                }
            ]
        )
    return stitched


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
