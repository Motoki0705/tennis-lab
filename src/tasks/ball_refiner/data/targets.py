"""Observed-only, single-ball amodal targets from the immutable ball store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    POINT_KIND_CODES,
    BallFrameStore,
    ClipRecord,
)
from src.tasks.ball_refiner.refiner_2d.contracts import Refiner2DTarget


class TargetReason(IntEnum):
    OBSERVED = 0
    OUT_OF_FRAME = 1
    UNREVIEWED = 2
    NO_INSTANCE = 3
    MULTIPLE_INSTANCES = 4
    INTERPOLATED = 5
    OCCLUSION_ESTIMATED = 6
    UNRESOLVED = 7


@dataclass(frozen=True)
class ClipTargets:
    """Every source frame, including unknowns; UV uses source (W-1,H-1).

    Estimated positions remain in ``uv`` for separate reference evaluation,
    but only OBSERVED frames have ``position_valid``. No target enters the
    model input. PTS and frame indices are retained without resampling.
    """

    frame_index: NDArray[np.int32]
    pts: NDArray[np.int64]
    timestamps_seconds: NDArray[np.float32]
    uv: NDArray[np.float32]
    position_valid: NDArray[np.bool_]
    presence: NDArray[np.bool_]
    presence_valid: NDArray[np.bool_]
    reason: NDArray[np.uint8]
    segment_break: NDArray[np.bool_]
    event: NDArray[np.uint8]

    def target(self, start: int, stop: int) -> Refiner2DTarget:
        """An unpadded window, with a singleton batch axis."""
        if not 0 <= start < stop <= len(self.pts):
            raise ValueError("Target window is outside the source timeline")
        window = slice(start, stop)
        return Refiner2DTarget(
            uv=torch.from_numpy(self.uv[window].copy())[None],
            position_valid=torch.from_numpy(self.position_valid[window].copy())[None],
            presence=torch.from_numpy(self.presence[window].copy())[None],
            presence_valid=torch.from_numpy(self.presence_valid[window].copy())[None],
            weight=torch.from_numpy(self.presence_valid[window].astype(np.float32))[None],
        )

    def counts(self) -> dict[str, int]:
        return {reason.name.lower(): int((self.reason == reason).sum()) for reason in TargetReason}


def project_store_targets(store: BallFrameStore, clip: ClipRecord) -> ClipTargets:
    """Do not reuse the detector's visible-ball negative-label policy.

    Empty reviewed frames are unknown amodal presence. Only exactly one
    explicitly out-of-frame ball is absent. Multiple instances, including
    one observed plus one unresolved/out-of-frame, are ambiguous and excluded.
    """
    if store.clip_by_id(clip.clip_id) != clip:
        raise ValueError("Clip does not belong to the declared store")
    if min(clip.source_width, clip.source_height) <= 1:
        raise ValueError("Source size must support endpoint normalization")
    rows = store.clip_rows(clip)
    n = len(rows)
    uv: NDArray[np.float32] = np.full((n, 2), np.nan, np.float32)
    reason: NDArray[np.uint8] = np.full(n, TargetReason.UNREVIEWED, np.uint8)
    kind_reason = {
        POINT_KIND_CODES[name.name.lower()]: name for name in (
            TargetReason.OBSERVED, TargetReason.OUT_OF_FRAME, TargetReason.INTERPOLATED,
            TargetReason.OCCLUSION_ESTIMATED, TargetReason.UNRESOLVED,
        )
    }
    # Undo the store's width-ratio resize BEFORE source endpoint normalization.
    denominator = clip.scale * np.asarray((clip.source_width - 1, clip.source_height - 1), np.float32)
    for i, row in enumerate(rows):
        if not store.frames["annotated"][row]:
            continue
        instances = store.instances_of(int(row))
        count = len(instances.point_kind)
        if count == 0:
            reason[i] = TargetReason.NO_INSTANCE
        elif count > 1:
            reason[i] = TargetReason.MULTIPLE_INSTANCES
        else:
            reason[i] = kind_reason[int(instances.point_kind[0])]
            uv[i] = instances.xy[0] / denominator
    located = np.isin(reason, [TargetReason.OBSERVED, TargetReason.INTERPOLATED, TargetReason.OCCLUSION_ESTIMATED])
    if not np.isfinite(uv[located]).all() or ((uv[located] < 0) | (uv[located] > 1)).any():
        raise ValueError(f"{clip.clip_id}: located labels exceed the source endpoint grid; do not clip them silently")
    pts = store.frames["pts"][rows].copy()
    # Subtract the first integer PTS before float32 conversion to preserve gaps.
    timestamps = ((pts - pts[0]).astype(np.float64) * float(Fraction(clip.time_base))).astype(np.float32)
    if not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
        raise ValueError(f"{clip.clip_id}: PTS cannot form strictly increasing float32 seconds")
    observed = reason == TargetReason.OBSERVED
    known = observed | (reason == TargetReason.OUT_OF_FRAME)
    return ClipTargets(
        frame_index=store.frames["frame_index"][rows].copy(), pts=pts, timestamps_seconds=timestamps,
        uv=uv, position_valid=observed, presence=observed.copy(), presence_valid=known, reason=reason,
        segment_break=store.frames["segment_break"][rows].copy(), event=store.frames["event"][rows].copy(),
    )
