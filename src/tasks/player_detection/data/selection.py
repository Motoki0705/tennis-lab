"""One frame eligibility decision shared by training and dataset inspection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.player_detection.configuration import FrameSelectionConfig

from .store import BBOX_SOURCE_CODES, PlayerFrameStore


@dataclass(frozen=True, slots=True)
class FrameEligibility:
    reason: str
    boxes_xyxy: NDArray[np.float32]
    dropped_small_instances: int = 0


def visible_boxes(
    boxes_xyxy: NDArray[np.float32], *, width: int, height: int, min_side_px: float
) -> tuple[NDArray[np.float32], int]:
    """Image-clipped amodal boxes; discarded instances never imply absence."""
    clipped = boxes_xyxy.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0.0, float(width))
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0.0, float(height))
    sides = np.minimum(clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1])
    keep = sides >= min_side_px
    return clipped[keep], int((~keep).sum())


def frame_eligibility(store: PlayerFrameStore, row: int, selection: FrameSelectionConfig) -> FrameEligibility:
    empty: NDArray[np.float32] = np.empty((0, 4), dtype=np.float32)
    if not store.frames["is_target"][row]:
        return FrameEligibility("context", empty)
    if selection.require_reviewed and not store.frames["reviewed"][row]:
        return FrameEligibility("unreviewed", empty)
    instances = store.instances_of(row)
    if len(instances.track_index) == 0:
        reason = "eligible" if store.frames["confirmed_empty"][row] else "unconfirmed_empty"
        return FrameEligibility(reason, empty)
    unresolved = instances.bbox_source == BBOX_SOURCE_CODES["unresolved"]
    if selection.require_located_players and unresolved.any():
        return FrameEligibility("unresolved_player", empty)
    clip = store.clip_of(row)
    boxes, dropped = visible_boxes(instances.boxes_xyxy[~unresolved], width=clip.width, height=clip.height,
        min_side_px=selection.min_visible_box_px)
    return FrameEligibility("eligible" if len(boxes) else "no_visible_box", boxes, dropped)
