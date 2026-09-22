"""One full-clip inference timeline and conservative source-frame ID restoration."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable


def association_frame_indices(
    frames: int, fps: float, *, target_fps: float = 30.0, max_frames: int = 1024
) -> NDArray[np.int64]:
    if frames < 1 or not math.isfinite(fps) or fps < 28 or target_fps != 30.0 or max_frames < 1:
        raise ValueError("Association requires nonempty video at >=28fps and a 30fps target")
    rate = min(target_fps, fps)
    grid: NDArray[np.float64] = np.arange(math.ceil(frames * rate / fps) + 1, dtype=np.float64)
    indices = np.unique(np.rint(grid * fps / rate).astype(np.int64))
    indices = indices[indices < frames]
    if len(indices) > max_frames:
        raise ReconstructionUnavailable("clip_too_long", f"Full clip has {len(indices)} association frames; maximum is {max_frames}")
    return indices


def restore_source_ids(
    uv: NDArray[np.float32],
    visibility: NDArray[np.bool_],
    sampled_ids: NDArray[np.int64],
    frame_indices: NDArray[np.int64],
    *,
    tracking: ObservationTrackingConfig,
) -> NDArray[np.int64]:
    """Only two accepted anchors and unique geometric matches can propagate an ID."""
    if uv.ndim != 5 or visibility.shape != uv.shape[:-1] or sampled_ids.shape != (uv.shape[0], len(frame_indices), uv.shape[2]):
        raise ValueError("ID restoration observation shapes disagree")
    if frame_indices.ndim != 1 or not len(frame_indices) or frame_indices[0] < 0 or frame_indices[-1] >= uv.shape[1] or (np.diff(frame_indices) <= 0).any():
        raise ValueError("Invalid source frame mapping")
    ids = np.full(uv.shape[:3], -1, np.int64)
    ids[:, frame_indices] = sampled_ids
    for left_index in range(len(frame_indices) - 1):
        left, right = int(frame_indices[left_index]), int(frame_indices[left_index + 1])
        for view in range(uv.shape[0]):
            left_ids, right_ids = sampled_ids[view, left_index], sampled_ids[view, left_index + 1]
            for values in (left_ids, right_ids):
                selected_ids = values[values >= 0]
                if len(np.unique(selected_ids)) != len(selected_ids):
                    raise ValueError("Sampled IDs must be unique per view/frame")
            common = sorted(set(left_ids[left_ids >= 0]) & set(right_ids[right_ids >= 0]))
            if not common:
                continue
            lrows = np.array([np.flatnonzero(left_ids == identity)[0] for identity in common])
            rrows = np.array([np.flatnonzero(right_ids == identity)[0] for identity in common])
            anchor_visible = visibility[view, left, lrows] & visibility[view, right, rrows]
            for frame in range(left + 1, right):
                carriers = np.flatnonzero(visibility[view, frame].any(-1))
                if not len(carriers):
                    continue
                alpha = (frame - left) / (right - left)
                predicted = (1 - alpha) * uv[view, left, lrows] + alpha * uv[view, right, rrows]
                cost: NDArray[np.float64] = np.full((len(common), len(carriers) + len(common)), tracking.max_distance, np.float64)
                for row in range(len(common)):
                    for column, carrier in enumerate(carriers):
                        shared = anchor_visible[row] & visibility[view, frame, carrier]
                        if shared.sum() < tracking.min_common_keypoints:
                            cost[row, column] = np.inf
                            continue
                        distances = np.linalg.norm(predicted[row, shared] - uv[view, frame, carrier, shared], axis=-1)
                        distance = np.median(distances) if tracking.cost_reduction == "median" else distances.mean()
                        cost[row, column] = float(distance) if distance < tracking.max_distance else np.inf
                rows, columns = linear_sum_assignment(cost)
                best = float(cost[rows, columns].sum())
                for row, column in zip(rows, columns, strict=True):
                    if column >= len(carriers):
                        continue
                    alternative = cost.copy()
                    alternative[row, column] = np.inf
                    ar, ac = linear_sum_assignment(alternative)
                    if float(alternative[ar, ac].sum()) - best <= 1e-6:
                        continue
                    ids[view, frame, carriers[column]] = common[row]
    return ids
