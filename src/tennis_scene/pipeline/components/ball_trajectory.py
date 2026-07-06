"""2D ball trajectory preparation for tennis-scene BLCS inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BallTrajectoryClip:
    """Completed 2D ball trajectory and its BLCS-valid frame range."""

    ball_uv: NDArray[np.float32]
    ball_mask: NDArray[np.bool_]
    start_frame: int
    end_frame: int


def complete_ball_trajectory_clip(
    ball_uv: NDArray[np.float32],
    visibility: NDArray[np.bool_],
) -> BallTrajectoryClip:
    """Interpolate detector observations into a contiguous BLCS input clip.

    Detector ``visibility`` means "observed by this camera at this frame", not
    "the 3D trajectory exists".  BLCS consumes a trajectory clip where every
    frame inside the clip is valid, so short 2D observation gaps are completed
    here before 3D lifting.
    """
    if ball_uv.ndim != 3 or ball_uv.shape[-1] != 2:
        raise ValueError(f"ball_uv must have shape (N, T, 2), got {ball_uv.shape}")
    num_cameras, num_frames = ball_uv.shape[:2]
    if visibility.shape != (num_cameras, num_frames):
        raise ValueError(
            "visibility must have shape (N, T), "
            f"got {visibility.shape} for {(num_cameras, num_frames)}"
        )

    finite_uv = np.isfinite(ball_uv).all(axis=-1)
    observed = visibility.astype(bool) & finite_uv
    observed_frames = observed.any(axis=0)
    if not observed_frames.any():
        raise ValueError("Cannot build a BLCS ball trajectory clip with no 2D observations.")

    observed_frame_indices = np.flatnonzero(observed_frames)
    start_frame = int(observed_frame_indices[0])
    end_frame = int(observed_frame_indices[-1]) + 1
    clip_indices: NDArray[np.float32] = np.arange(
        start_frame,
        end_frame,
        dtype=np.float32,
    )

    completed = np.asarray(ball_uv, dtype=np.float32).copy()
    ball_mask = np.zeros((num_cameras, num_frames), dtype=np.bool_)

    for camera_index in range(num_cameras):
        camera_observed_indices = np.flatnonzero(observed[camera_index])
        camera_clip_indices = camera_observed_indices[
            (camera_observed_indices >= start_frame)
            & (camera_observed_indices < end_frame)
        ]
        if camera_clip_indices.size == 0:
            raise ValueError(
                "Cannot complete BLCS ball trajectory clip because camera "
                f"{camera_index} has no finite 2D ball observations in "
                f"[{start_frame}, {end_frame})."
            )

        camera_observed_uv = completed[camera_index, camera_clip_indices]
        if camera_clip_indices.size == 1:
            interpolated_uv = np.repeat(
                camera_observed_uv,
                repeats=end_frame - start_frame,
                axis=0,
            )
        else:
            sample_indices = camera_clip_indices.astype(np.float32)
            interpolated_u = np.interp(
                clip_indices,
                sample_indices,
                camera_observed_uv[:, 0],
            )
            interpolated_v = np.interp(
                clip_indices,
                sample_indices,
                camera_observed_uv[:, 1],
            )
            interpolated_uv = np.stack([interpolated_u, interpolated_v], axis=-1)

        completed[camera_index, start_frame:end_frame] = interpolated_uv.astype(
            np.float32
        )
        ball_mask[camera_index, start_frame:end_frame] = True

    return BallTrajectoryClip(
        ball_uv=completed,
        ball_mask=ball_mask,
        start_frame=start_frame,
        end_frame=end_frame,
    )


__all__ = ["BallTrajectoryClip", "complete_ball_trajectory_clip"]
