"""Saved single-object observations and diagnostics, without training transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.base.visualization.review.camera import CameraRecord
from src.utils.projection.camera_projector import project_points
from src.utils.schema.court import COURT_KP_NAMES
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
)

NORMALIZATION_ATOL = 1.0e-4
EVENT_FIELDS = ("t_start", "t_net", "t_bounce1", "t_bounce2", "t_bounce3", "t_return")


def load_array(
    directory: Path, name: str, shape: tuple[int, ...], *, mask: bool = False
) -> np.ndarray | None:
    """Missing optional evidence stays missing; malformed arrays fail explicitly."""
    path = directory / f"{name}.npy"
    if not path.is_file():
        return None
    value: np.ndarray = np.load(path, allow_pickle=False)
    if value.shape != shape:
        raise ValueError(f"{path.name}: expected shape {shape}, got {value.shape}.")
    if mask:
        if value.dtype != np.bool_:
            raise ValueError(f"{path.name}: visibility must have bool dtype.")
    elif not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{path.name}: coordinates must have floating dtype.")
    return value


def json_array(value: np.ndarray | None) -> Any:
    """Represent non-finite coordinates as null, never as a synthetic zero."""
    if value is None:
        return None
    if value.dtype == np.bool_:
        return value.tolist()
    clean: np.ndarray = value.astype(object)
    clean[~np.isfinite(value)] = None
    return clean.tolist()


def normalization_diagnostic(
    physical: np.ndarray | None, normalized: np.ndarray | None, *, velocity: bool
) -> dict[str, Any]:
    """Compare the independently saved pair after the canonical unit conversion."""
    if physical is None or normalized is None:
        return {
            "status": "missing",
            "max_abs_error": None,
            "tolerance": NORMALIZATION_ATOL,
        }
    if physical.shape != normalized.shape:
        raise ValueError(
            "Saved physical and normalized arrays must have the same shape."
        )
    restored = (
        denormalize_court_velocity(normalized)
        if velocity
        else denormalize_court_position(normalized)
    )
    if not np.isfinite(physical).all() or not np.isfinite(restored).all():
        return {
            "status": "nonfinite",
            "max_abs_error": None,
            "tolerance": NORMALIZATION_ATOL,
        }
    error = float(np.max(np.abs(physical - restored)))
    return {
        "status": "ok" if error <= NORMALIZATION_ATOL else "mismatch",
        "max_abs_error": error,
        "tolerance": NORMALIZATION_ATOL,
    }


def _projection(
    camera: CameraRecord, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    uv, in_front = project_points(
        camera.to_camera(), torch.as_tensor(points, dtype=torch.float32)
    )
    coordinates = uv.numpy() / np.asarray(camera.image_size, dtype=np.float32)
    front = in_front.numpy() & np.isfinite(points).all(axis=-1)
    # The production projector supplies safe arithmetic for points behind the
    # camera. Those UVs are not meaningful image observations in the reviewer.
    coordinates[~front] = np.nan
    return coordinates, front


def _comparison(
    camera: CameraRecord,
    projected: np.ndarray,
    front: np.ndarray,
    saved: np.ndarray | None,
    visibility: np.ndarray | None,
) -> dict[str, Any]:
    visible = (
        front
        & np.isfinite(projected).all(axis=-1)
        & (projected >= 0).all(axis=-1)
        & (projected <= 1).all(axis=-1)
    )
    error = None
    if saved is not None:
        comparable = (
            front
            & np.isfinite(projected).all(axis=-1)
            & np.isfinite(saved).all(axis=-1)
        )
        error = np.full(projected.shape[0], np.nan, dtype=np.float32)
        error[comparable] = np.linalg.norm(
            (saved[comparable] - projected[comparable]) * np.asarray(camera.image_size),
            axis=-1,
        )
    finite_errors = error[np.isfinite(error)] if error is not None else np.array([])
    return {
        "saved_uv": json_array(saved),
        "saved_visibility": json_array(visibility),
        "projected_uv": json_array(projected),
        "in_front": front.tolist(),
        "expected_visibility": visible.tolist(),
        "error_px": json_array(error),
        "summary": {
            "saved_visible": int(visibility.sum()) if visibility is not None else None,
            "total": int(projected.shape[0]),
            "invalid_saved_uv": int((~np.isfinite(saved).all(axis=-1)).sum())
            if saved is not None
            else None,
            "comparable": int(finite_errors.size),
            "max_error_px": float(finite_errors.max()) if finite_errors.size else None,
            "visibility_mismatches": int((visibility != visible).sum())
            if visibility is not None
            else None,
        },
    }


def camera_observations(
    directory: Path, camera: CameraRecord, positions: np.ndarray, court: np.ndarray
) -> dict[str, Any]:
    """Keep saved UV/visibility separate from a projection of the metre teachers."""
    prefix = camera.id
    count = int(positions.shape[0])
    ball_uv = load_array(directory, f"{prefix}_ball_uv", (count, 2))
    ball_vis = load_array(directory, f"{prefix}_ball_vis", (count,), mask=True)
    court_uv = load_array(directory, f"{prefix}_court_kp_uv", (20, 2))
    court_vis = load_array(directory, f"{prefix}_court_kp_vis", (20,), mask=True)
    ball_projected, ball_front = _projection(camera, positions)
    court_projected, court_front = _projection(camera, court)
    return {
        "id": camera.id,
        "image_size": list(camera.image_size),
        "center_m": list(camera.center),
        "rotation_world_to_camera": [list(row) for row in camera.rotation],
        "intrinsics_px": camera.intrinsics().tolist(),
        "ball": _comparison(camera, ball_projected, ball_front, ball_uv, ball_vis),
        "court": _comparison(camera, court_projected, court_front, court_uv, court_vis),
    }


def shot_events(
    meta: dict[str, Any], *, frames: int, fps: float
) -> list[dict[str, Any]] | None:
    """Classify saved output-frame events against their actual retained shot.

    Each shot was simulated past its sampled return before concatenation. Its
    later bounce metadata can therefore describe a discarded continuation.
    Preserve those times but do not put them on the active event timeline.
    """
    raw = meta.get("shots")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("meta.shots must be an array.")
    starts: list[int] = []
    for shot in raw:
        if not isinstance(shot, dict) or type(shot.get("t_start")) is not int:
            raise ValueError("Every shot requires an integer t_start.")
        starts.append(shot["t_start"])
    if any(start < 0 or start >= frames for start in starts) or any(
        a >= b for a, b in zip(starts, starts[1:], strict=False)
    ):
        raise ValueError("Shot starts must increase within the saved frame range.")
    events = []
    for index, shot in enumerate(raw):
        end = starts[index + 1] if index + 1 < len(starts) else frames
        for field in EVENT_FIELDS:
            frame = shot.get(field)
            if frame is None:
                status = "missing"
            elif type(frame) is not int or frame < -1:
                raise ValueError(
                    f"shots[{index}].{field} must be an output frame or -1."
                )
            elif frame == -1:
                status = "not_recorded"
            elif frame >= frames:
                status = "outside_scene"
            elif frame < starts[index]:
                status = "before_shot"
            elif frame > end or (frame == end and field != "t_return"):
                status = "after_shot"
            else:
                status = "on_trajectory"
            events.append(
                {
                    "shot_record_index": index,
                    "shot_index": shot.get("shot_index"),
                    "field": field,
                    "kind": "hit" if field == "t_start" else field.removeprefix("t_"),
                    "frame": frame,
                    "time_seconds": frame / fps
                    if type(frame) is int and frame >= 0
                    else None,
                    "status": status,
                    "from_side": shot.get("from_side"),
                    "shot_type": shot.get("shot_type"),
                    "return_type": shot.get("return_type"),
                }
            )
    return events


def load_inspection(
    directory: Path,
    document: dict[str, Any],
    meta: dict[str, Any],
    cameras: tuple[CameraRecord, ...],
) -> dict[str, Any]:
    """Assemble JSON evidence for the same revision as the shared 3D buffer."""
    frames = int(document["frame_count"])
    shape = (frames, 3)
    position = load_array(directory, "ball_pos_world", shape)
    if position is None or not np.isfinite(position).all():
        raise ValueError(
            "ball_pos_world.npy must contain finite metre-valued positions."
        )
    velocity = load_array(directory, "ball_vel_world", shape)
    position_norm = load_array(directory, "ball_pos_norm", shape)
    velocity_norm = load_array(directory, "ball_vel_norm", shape)
    return {
        "schema": "blcs_dataset_inspection_v1",
        "scene_id": document["scene_id"],
        "form": document["form"],
        "revision": document["revision"],
        "frame_count": frames,
        "fps": document["fps"],
        "source": {
            "kind": "physical_simulation",
            "rgb": "not_saved",
            "path": f"blcs/single_object/scenes/{directory.name}",
        },
        "contract": {
            "court_selector": "physical_v1",
            "court_saved_points": 20,
            "court_keypoint_names": list(COURT_KP_NAMES),
            "coordinate_frame": document["coordinate_frame"],
            "normalization": meta["court_coordinate_normalization"],
            "uv_units": "u / image_width, v / image_height",
            "time_source": "output frame index / fps_out (no saved timestamp array)",
        },
        "ball": {
            "position_m": json_array(position),
            "velocity_mps": json_array(velocity),
            "position_normalized": json_array(position_norm),
            "velocity_normalized": json_array(velocity_norm),
            "normalization": {
                "position": normalization_diagnostic(
                    position, position_norm, velocity=False
                ),
                "velocity": normalization_diagnostic(
                    velocity, velocity_norm, velocity=True
                ),
            },
        },
        "cameras": [
            camera_observations(
                directory,
                camera,
                position,
                np.asarray(document["court"]["keypoints"], dtype=np.float32),
            )
            for camera in cameras
        ],
        "events": shot_events(meta, frames=frames, fps=float(document["fps"])),
        "rally": {
            key: meta.get(key) for key in ("rally_length", "end_reason", "winner_side")
        },
    }


__all__ = [
    "camera_observations",
    "load_inspection",
    "normalization_diagnostic",
    "shot_events",
]
