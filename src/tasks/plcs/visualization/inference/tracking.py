"""Single-scene PLCS track-query batch assembly for the inference UI.

The track-query model does not accept a loaded scene directly.  Its inputs are
the post-corruption observations that ``PLCSTrackingDataset`` produces *after*
observation association and lifecycle slot packing, so the UI reuses that very
dataset with the request's window and cameras fixed in place instead of
re-deriving the pipeline.  The assembled batch is the same tensor contract the
trainer uses, which keeps the inference path honest.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.base.data.scene_dataset import (
    CameraSelection,
    Scene,
    TemporalWindow,
)
from src.tasks.base.training.tracking_metrics import TrackingMetricConfig
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)
from src.tasks.plcs.model_io import (
    PLCSReferenceMetadata,
    plcs_reference_metadata_from_batch,
)


class TrackingSceneError(ValueError):
    """Raised when a tracking window cannot be assembled into a batch."""


@dataclass(frozen=True, slots=True)
class TrackingWindow:
    """One fixed scene window and camera subset for track-query inference."""

    family_dir: Path
    scene_id: str
    cameras: tuple[int, ...]
    start: int
    length: int
    reference_camera_id: str | None


class SingleWindowTrackingDataset(PLCSTrackingDataset):
    """``PLCSTrackingDataset`` whose window and cameras are request-fixed.

    Only ``select_window``/``select_cameras`` are overridden; observation
    tracking, court alignment, reference metadata, and lifecycle packing all
    stay on the shared ``build_sample``/``augment_sample`` path.
    """

    def __init__(self, *, window: TrackingWindow, **kwargs: Any) -> None:
        self._window = window
        super().__init__(
            reference_camera_id=window.reference_camera_id,
            **kwargs,
        )

    def select_window(
        self,
        scene: Scene,
        *,
        full_len: int | None = None,
        seq_len_range: tuple[int, int] | None = None,
        crop_mode: str | None = None,
    ) -> TemporalWindow:
        del full_len, seq_len_range, crop_mode
        window = self._window
        return TemporalWindow(
            start=window.start,
            end=window.start + window.length,
            seq_len=window.length,
            full_len=int(scene.num_frames),
        )

    def select_cameras(
        self,
        scene: Scene,
        *,
        num_views_range: tuple[int, int] | None = None,
        camera_mode: str | int | None = None,
    ) -> CameraSelection:
        del scene, num_views_range, camera_mode
        return CameraSelection(indices=tuple(self._window.cameras))


def tracking_metric_config(config: Mapping[str, Any]) -> TrackingMetricConfig:
    """Return the checkpoint's lifecycle-metric thresholds (no defaults)."""
    raw = config.get("tracking_metrics")
    if not isinstance(raw, Mapping):
        raise TrackingSceneError(
            "track-query checkpoint config has no tracking_metrics block; "
            "refusing to invent a presence threshold."
        )
    return TrackingMetricConfig.from_mapping(raw)


def build_tracking_batch(
    *,
    config: Mapping[str, Any],
    window: TrackingWindow,
) -> dict[str, Any]:
    """Assemble one collated ``(B=1)`` track-query batch from a real window."""
    with TemporaryDirectory(prefix="plcs-inference-ui-") as scratch:
        split_file = Path(scratch) / "window.txt"
        split_file.write_text(f"{window.scene_id}\n", encoding="utf-8")
        try:
            dataset = SingleWindowTrackingDataset(
                scene_dir=window.family_dir,
                split_file=split_file,
                config=config,
                augment=False,
                window=window,
            )
        except ValueError as error:
            raise TrackingSceneError(
                f"このチェックポイントの config では scene を読み込めません: {error}"
            ) from error
        if len(dataset) != 1:
            raise TrackingSceneError(
                "the fixed window split must contain exactly one scene, got "
                f"{len(dataset)}."
            )
        try:
            sample = cast("dict[str, Any]", dataset[0])
        except ValueError as error:
            raise TrackingSceneError(
                "この窓では lifecycle slot を構成できません (人数が model.num_queries"
                f" を超えている可能性があります): {error}"
            ) from error
    collated: dict[str, Any] = collate_plcs_tracking_batch([sample])
    return collated


def reference_metadata_from_batch(
    batch: Mapping[str, Any],
) -> PLCSReferenceMetadata | None:
    """Return the typed track-query reference metadata, or ``None`` for v1."""
    return plcs_reference_metadata_from_batch(batch)


@dataclass(frozen=True, slots=True)
class TrackMatch:
    """Per-frame Hungarian matching between predicted slots and GT tracks."""

    matched_frames: int
    matched_pairs: int
    unmatched_prediction: int
    unmatched_ground_truth: int
    position_error_m: NDArray[np.float64]
    yaw_error_deg: NDArray[np.float64]


def match_tracks(
    *,
    pred_position_m: NDArray[np.float64],
    pred_present: NDArray[np.bool_],
    pred_rotation: NDArray[np.float64] | None,
    gt_position_m: NDArray[np.float64],
    gt_present: NDArray[np.bool_],
    gt_rotation: NDArray[np.float64] | None,
    max_distance_m: float | None = None,
) -> TrackMatch:
    """Match each frame one-to-one with the shared Hungarian primitive.

    Naive index-wise comparison of query slots and physical tracks is not a
    valid multi-object metric, so every frame is matched with
    :func:`scipy.optimize.linear_sum_assignment` over physical metre positions.
    ``max_distance_m=None`` keeps the minimum-cost assignment ungated, giving
    the best achievable per-frame pairing; a positive value rejects assignments
    farther than the gate.
    """
    pred_position = np.asarray(pred_position_m, dtype=np.float64)
    gt_position = np.asarray(gt_position_m, dtype=np.float64)
    pred_mask = np.asarray(pred_present, dtype=bool)
    gt_mask = np.asarray(gt_present, dtype=bool)
    if (
        pred_position.ndim != 3
        or gt_position.ndim != 3
        or pred_position.shape[-1] != 3
        or gt_position.shape[-1] != 3
    ):
        raise TrackingSceneError(
            "track positions must have shape (T, N, 3), got "
            f"{pred_position.shape} and {gt_position.shape}."
        )
    frames = int(pred_position.shape[0])
    if gt_position.shape[0] != frames:
        raise TrackingSceneError(
            "prediction and ground-truth track counts must share the time axis: "
            f"{pred_position.shape} vs {gt_position.shape}."
        )
    if (
        pred_mask.shape != pred_position.shape[:2]
        or gt_mask.shape != (gt_position.shape[:2])
    ):
        raise TrackingSceneError(
            "presence masks must match their position track axes: "
            f"{pred_mask.shape}/{gt_mask.shape} vs "
            f"{pred_position.shape[:2]}/{gt_position.shape[:2]}."
        )
    if max_distance_m is not None and (
        max_distance_m <= 0.0 or not math.isfinite(max_distance_m)
    ):
        raise TrackingSceneError("max_distance_m must be positive and finite.")
    if frames == 0:
        raise TrackingSceneError("track matching requires at least one frame.")

    pred_rotation = (
        None if pred_rotation is None else np.asarray(pred_rotation, dtype=np.float64)
    )
    gt_rotation = (
        None if gt_rotation is None else np.asarray(gt_rotation, dtype=np.float64)
    )
    if pred_rotation is not None and pred_rotation.shape != (
        *pred_position.shape[:-1],
        2,
    ):
        raise TrackingSceneError(
            f"prediction rotation must match (T, Q, 2), got {pred_rotation.shape}."
        )
    if gt_rotation is not None and gt_rotation.shape != (*gt_position.shape[:-1], 2):
        raise TrackingSceneError(
            f"ground-truth rotation must match (T, P, 2), got {gt_rotation.shape}."
        )
    position_errors: list[float] = []
    yaw_errors: list[float] = []
    matched_pairs = 0
    unmatched_prediction = 0
    unmatched_ground_truth = 0
    matched_frames = 0
    for frame in range(frames):
        pred_indices = np.flatnonzero(pred_mask[frame])
        gt_indices = np.flatnonzero(gt_mask[frame])
        if pred_indices.size == 0 and gt_indices.size == 0:
            continue
        matched_frames += 1
        if pred_indices.size == 0 or gt_indices.size == 0:
            unmatched_prediction += int(pred_indices.size)
            unmatched_ground_truth += int(gt_indices.size)
            continue
        # Index the frame first so the advanced indices broadcast as
        # (Q_active, 1, 3) and (1, P_active, 3) rather than axis-swapped.
        pred_frame = pred_position[frame][pred_indices, None, :]
        gt_frame = gt_position[frame][None, gt_indices, :]
        costs = np.linalg.norm(pred_frame - gt_frame, axis=-1)
        rows, columns = linear_sum_assignment(costs)
        if max_distance_m is None:
            accepted = np.ones(rows.shape, dtype=bool)
        else:
            accepted = costs[rows, columns] <= max_distance_m
        # ``linear_sum_assignment`` returns ``min(|Q|, |P|)`` pairs, so the
        # larger side's surplus is always unmatched regardless of the gate.
        matched_pairs += int(accepted.sum())
        unmatched_prediction += int(pred_indices.size - accepted.sum())
        unmatched_ground_truth += int(gt_indices.size - accepted.sum())
        for row, column, keep in zip(rows, columns, accepted, strict=True):
            if not keep:
                continue
            pred_index = int(pred_indices[row])
            gt_index = int(gt_indices[column])
            position_errors.append(float(costs[row, column]))
            if pred_rotation is not None and gt_rotation is not None:
                yaw = _yaw_error_degrees(
                    gt_rotation[frame, gt_index],
                    pred_rotation[frame, pred_index],
                )
                if yaw is not None:
                    yaw_errors.append(yaw)
    return TrackMatch(
        matched_frames=matched_frames,
        matched_pairs=matched_pairs,
        unmatched_prediction=unmatched_prediction,
        unmatched_ground_truth=unmatched_ground_truth,
        position_error_m=np.asarray(position_errors, dtype=np.float64),
        yaw_error_deg=np.asarray(yaw_errors, dtype=np.float64),
    )


def _yaw_error_degrees(
    gt: NDArray[np.float64], pred: NDArray[np.float64]
) -> float | None:
    if gt.shape != (2,) or pred.shape != (2,):
        return None
    gt_norm = float(np.linalg.norm(gt))
    pred_norm = float(np.linalg.norm(pred))
    if gt_norm == 0.0 or pred_norm == 0.0:
        return None
    gt_unit = gt / gt_norm
    pred_unit = pred / pred_norm
    cross = gt_unit[0] * pred_unit[1] - gt_unit[1] * pred_unit[0]
    dot = gt_unit[0] * pred_unit[0] + gt_unit[1] * pred_unit[1]
    return float(abs(np.degrees(np.arctan2(cross, dot))))


__all__ = [
    "SingleWindowTrackingDataset",
    "TrackMatch",
    "TrackingSceneError",
    "TrackingWindow",
    "build_tracking_batch",
    "match_tracks",
    "reference_metadata_from_batch",
    "tracking_metric_config",
]
