"""Court-space landmark distances and motion supported by observed samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.schema.player import COCO_KP_IDX

PLAYER_LANDMARKS = ("root", "left_wrist", "right_wrist")
_WRISTS = [COCO_KP_IDX[name] for name in PLAYER_LANDMARKS[1:]]


@dataclass(frozen=True, slots=True)
class PlayerEvidence:
    """Per-player root and both court-space COCO17 wrists, in that order."""

    positions: NDArray[np.float32]  # (P, T, 3 landmarks, XYZ)
    valid: NDArray[np.bool_]  # (P, T, 3 landmarks)
    track_ids: NDArray[np.int32] | None  # (P,), local to this clip

    @classmethod
    def from_pose(
        cls,
        roots: NDArray[np.float32],
        root_valid: NDArray[np.bool_],
        joints: NDArray[np.float32],
        joint_valid: NDArray[np.bool_],
        track_ids: NDArray[np.int32] | None,
    ) -> PlayerEvidence:
        return cls(
            np.concatenate((roots[:, :, None, :], joints[:, :, _WRISTS, :]), axis=2),
            np.concatenate(
                (root_valid[:, :, None], joint_valid[:, :, _WRISTS]), axis=2
            ),
            track_ids,
        )


def landmark_distances(
    positions: NDArray[np.float32],
    valid: NDArray[np.bool_],
    ball: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Measure metres along the final XYZ axis, with infinity for masked landmarks."""
    return np.where(valid, np.linalg.norm(positions - ball, axis=-1), np.inf)


def side_velocities(
    xyz: NDArray[np.float64], valid: NDArray[np.bool_], fps: float, window: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Shorten each secant at a contiguous run boundary. An empty side has NaN velocity.
    indices = np.arange(len(xyz))
    run_starts = np.maximum.accumulate(np.where(valid, 0, indices + 1))
    run_ends = np.minimum.accumulate(np.where(valid, len(xyz) - 1, indices - 1)[::-1])[
        ::-1
    ]
    incoming = np.full_like(xyz, np.nan)
    outgoing = np.full_like(xyz, np.nan)
    left = np.maximum(run_starts, indices - window)
    right = np.minimum(run_ends, indices + window)
    before = valid & (left < indices)
    after = valid & (right > indices)
    incoming[before] = (xyz[before] - xyz[left[before]]) * (
        fps / (indices[before] - left[before])
    )[:, None]
    outgoing[after] = (xyz[right[after]] - xyz[after]) * (
        fps / (right[after] - indices[after])
    )[:, None]
    return incoming, outgoing
