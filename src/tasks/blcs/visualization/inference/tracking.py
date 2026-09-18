"""Multi-object scene-to-input builder and matched metrics for the BLCS UI.

The multi-object (tracking) path in the inference UI must reuse the repository's
canonical tracking pipeline instead of re-implementing observation association,
reference selection, or query packing. This module therefore instantiates the
existing :class:`~src.tasks.blcs.data.tracking_dataset.BLCSTrackingDataset` from
the checkpoint's own training config, pins its window/camera selection to the
exact request through a thin subclass, and collates a single-scene batch with the
canonical :func:`collate_blcs_tracking_batch`.

Metrics never assume query slot ``i`` corresponds to ground-truth object ``i``:
present predictions and present ground-truth objects are matched per frame with a
minimum-cost (Hungarian) assignment before the position error is measured.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from src.tasks.base.data.scene_dataset import CameraSelection, Scene, TemporalWindow
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)

DEFAULT_POSITION_THRESHOLD_M = 0.3


@dataclass(frozen=True, slots=True)
class TrackingInput:
    """One collated single-scene tracking batch and its aligned ground truth."""

    batch: dict[str, Any]
    ground_truth: NDArray[np.float64]
    ground_truth_present: NDArray[np.bool_]
    window_start: int
    window_length: int
    scene_frames: int


class _FixedWindowTrackingDataset(BLCSTrackingDataset):
    """A tracking dataset pinned to one UI-selected window and camera subset."""

    def __init__(
        self,
        *,
        window: TemporalWindow,
        camera_indices: tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        self._fixed_window = window
        self._fixed_camera_indices = camera_indices
        super().__init__(**kwargs)

    def select_window(
        self,
        scene: Scene,
        *,
        full_len: int | None = None,
        seq_len_range: tuple[int, int] | None = None,
        crop_mode: str | None = None,
    ) -> TemporalWindow:
        """Return the request window verbatim instead of sampling one."""
        return self._fixed_window

    def select_cameras(
        self,
        scene: Scene,
        *,
        num_views_range: tuple[int, int] | None = None,
        camera_mode: str | int | None = None,
    ) -> CameraSelection:
        """Return the request camera subset verbatim instead of sampling one."""
        return CameraSelection(indices=self._fixed_camera_indices)


def _pin_config(
    config: Any,
    *,
    window_length: int,
    num_cameras: int,
    seed: int,
) -> Any:
    """Pin ``data.seq_len_range``/``data.num_views_range`` to the request.

    The canonical dataset filters scenes and samples windows from these ranges;
    replacing them with the exact request keeps the dataset's own validation and
    deterministic single-sample behaviour while matching what the UI asked for.
    """
    pinned = OmegaConf.create(copy.deepcopy(config))
    if not isinstance(pinned, DictConfig):
        raise TypeError("Tracking inference requires a mapping checkpoint config.")
    with open_dict(pinned):
        data = pinned["data"]
        data["seq_len_range"] = [int(window_length), int(window_length)]
        data["num_views_range"] = [int(num_cameras), int(num_cameras)]
        run = pinned.get("run")
        if run is None:
            pinned["run"] = {"seed": int(seed)}
        else:
            run["seed"] = int(seed)
    return pinned


def build_tracking_input(
    *,
    scene_dir: Path,
    scene_id: str,
    config: Any,
    reference_camera_id: str | None,
    camera_indices: Sequence[int],
    window_start: int,
    window_length: int,
    seed: int,
) -> TrackingInput:
    """Build one windowed tracking batch from the canonical scene dataset."""
    cameras = tuple(int(index) for index in camera_indices)
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    if window_length <= 0:
        raise ValueError("window_length must be positive.")
    pinned = _pin_config(
        config,
        window_length=window_length,
        num_cameras=len(cameras),
        seed=seed,
    )
    scene_path = Path(scene_dir) / "scenes" / scene_id
    if not scene_path.is_dir():
        raise FileNotFoundError(f"Scene directory is missing: {scene_path}")
    with tempfile.TemporaryDirectory(prefix="blcs_tracking_ui_") as temporary:
        split_file = Path(temporary) / "single_scene.txt"
        split_file.write_text(f"{scene_id}\n", encoding="utf-8")
        dataset = _FixedWindowTrackingDataset(
            scene_dir=str(scene_dir),
            split_file=str(split_file),
            config=pinned,
            seed=seed,
            augment=False,
            reference_camera_id=reference_camera_id,
            window=TemporalWindow(
                start=window_start,
                end=window_start + window_length,
                seq_len=window_length,
                full_len=window_start + window_length,
            ),
            camera_indices=cameras,
        )
        if len(dataset) != 1:
            raise ValueError(
                "Tracking inference expected exactly one scene, got "
                f"{len(dataset)}."
            )
        sample = dataset[0]
        batch = collate_blcs_tracking_batch([sample])
    positions = np.asarray(sample["target_position"], dtype=np.float64)
    present = np.asarray(sample["target_presence"], dtype=bool)
    if positions.ndim != 3 or positions.shape[:2] != present.shape:
        raise ValueError(
            "Tracking target tensors must share (T, tracks) presence axes."
        )
    return TrackingInput(
        batch=batch,
        ground_truth=positions,
        ground_truth_present=present,
        window_start=window_start,
        window_length=window_length,
        scene_frames=int(dataset.scene_headers[0].num_frames),
    )


def _match_frame(
    predicted: NDArray[np.float64],
    target: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return minimum-cost one-to-one query/target indices for one frame."""
    if predicted.shape[0] == 0 or target.shape[0] == 0:
        empty: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        return empty, empty
    distances = np.linalg.norm(
        predicted[:, None, :] - target[None, :, :],
        axis=-1,
    )
    rows, columns = linear_sum_assignment(distances)
    return rows.astype(np.int64), columns.astype(np.int64)


def tracking_metrics(
    *,
    predicted_position: NDArray[np.float64],
    predicted_present: NDArray[np.bool_],
    target_position: NDArray[np.float64],
    target_present: NDArray[np.bool_],
    position_threshold_m: float = DEFAULT_POSITION_THRESHOLD_M,
) -> dict[str, float | None]:
    """Compute matched multi-object localization and presence statistics.

    Ground-truth and predicted objects are matched per frame without assuming a
    shared slot index, so query permutation cannot inflate or deflate the error.
    Undefined ratios (for example when no ground-truth object is ever present)
    are reported as ``None`` so the payload stays strictly JSON-serialisable.
    """
    frames = int(predicted_position.shape[0])
    errors: list[NDArray[np.float64]] = []
    endpoint_errors: list[NDArray[np.float64]] = []
    matched_pairs = 0
    predicted_positive = 0
    target_positive = 0
    true_positive = 0
    for frame in range(frames):
        active_target = np.flatnonzero(target_present[frame])
        active_prediction = np.flatnonzero(predicted_present[frame])
        predicted_positive += int(active_prediction.size)
        target_positive += int(active_target.size)
        if active_target.size == 0 or active_prediction.size == 0:
            continue
        rows, columns = _match_frame(
            predicted_position[frame, active_prediction],
            target_position[frame, active_target],
        )
        frame_errors = np.linalg.norm(
            predicted_position[frame, active_prediction[rows]]
            - target_position[frame, active_target[columns]],
            axis=-1,
        )
        errors.append(frame_errors)
        matched_pairs += int(frame_errors.size)
        true_positive += int(frame_errors.size)
        if frame == frames - 1:
            endpoint_errors.append(frame_errors)
    if errors:
        stacked = np.concatenate(errors)
        position_error: float | None = float(stacked.mean())
        accuracy: float | None = float(np.mean(stacked <= position_threshold_m))
    else:
        position_error = None
        accuracy = None
    endpoint_error = (
        float(np.concatenate(endpoint_errors).mean())
        if endpoint_errors
        else None
    )
    precision = (
        true_positive / predicted_positive if predicted_positive else None
    )
    recall = true_positive / target_positive if target_positive else None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "position_error_m": position_error,
        "endpoint_error_m": endpoint_error,
        "accuracy_0p3m": accuracy,
        "matched_pairs": float(matched_pairs),
        "predicted_present": float(predicted_positive),
        "gt_present": float(target_positive),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
    }


def tracking_prediction_payload(
    position: Tensor,
    presence: Tensor,
) -> dict[str, Any]:
    """Flatten one decoded ``(T, tracks, 3)`` prediction for the browser."""
    if position.ndim != 4 or position.shape[-1] != 3 or position.shape[0] != 1:
        raise ValueError("Tracking prediction must have shape (1, T, tracks, 3).")
    if presence.shape != position.shape[:3]:
        raise ValueError("Tracking presence must match position without XYZ.")
    frames = int(position.shape[1])
    tracks = int(position.shape[2])
    return {
        "frames": frames,
        "tracks": tracks,
        "positions": [float(value) for value in position[0].reshape(-1).tolist()],
        "presence": [int(value) for value in presence[0].reshape(-1).tolist()],
    }


def position_threshold_from_config(config: Any, *, default: float) -> float:
    """Read ``metrics.position_threshold_m`` from a checkpoint config if present."""
    if not isinstance(config, (DictConfig, dict)):
        return default
    metrics = config.get("metrics") if hasattr(config, "get") else None
    if metrics is None:
        return default
    value = metrics.get("position_threshold_m")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        return default
    return float(value)


__all__ = [
    "DEFAULT_POSITION_THRESHOLD_M",
    "TrackingInput",
    "build_tracking_input",
    "position_threshold_from_config",
    "tracking_metrics",
    "tracking_prediction_payload",
]
