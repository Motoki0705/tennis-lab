"""Input shape normalization helpers for tennis scene pipeline components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_model_num_court_keypoints(model: object, fallback: int = 20) -> int:
    """Return the court keypoint count expected by a loaded model."""
    value = getattr(model, "num_court_tokens", fallback)
    return int(value)


def _as_float32(name: str, array: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(array, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_visibility(
    name: str,
    array: NDArray[np.float32] | None,
    shape: tuple[int, ...],
) -> NDArray[np.float32]:
    if array is None:
        return np.ones(shape, dtype=np.float32)
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _pad_court_keypoints(
    court_kp: NDArray[np.float32],
    court_vis: NDArray[np.float32],
    *,
    target_num_keypoints: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    num_keypoints = int(court_kp.shape[-2])
    if num_keypoints > target_num_keypoints:
        raise ValueError(
            "court_kp has more keypoints than the model expects: "
            f"K={num_keypoints}, expected K={target_num_keypoints}"
        )
    if num_keypoints == target_num_keypoints:
        return court_kp.astype(np.float32), court_vis.astype(np.float32)

    pad_count = target_num_keypoints - num_keypoints
    kp_pad_shape = (*court_kp.shape[:-2], pad_count, 2)
    vis_pad_shape = (*court_vis.shape[:-1], pad_count)
    return (
        np.concatenate(
            [court_kp, np.zeros(kp_pad_shape, dtype=np.float32)],
            axis=-2,
        ).astype(np.float32),
        np.concatenate(
            [court_vis, np.zeros(vis_pad_shape, dtype=np.float32)],
            axis=-1,
        ).astype(np.float32),
    )


def normalize_ball_uv_sequence(
    ball_uv: NDArray[np.float32],
    ball_vis: NDArray[np.bool_] | NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[int, int, int]]:
    """Normalize ball inputs to (B, N, T, 2) and visibility to (B, N, T)."""
    ball = _as_float32("ball_uv", ball_uv)
    if ball.ndim == 2 and ball.shape[-1] == 2:
        ball = ball[None, None, ...]
    elif ball.ndim == 3 and ball.shape[-1] == 2:
        ball = ball[None, ...]
    elif ball.ndim == 4 and ball.shape[-1] == 2:
        ball = ball
    else:
        raise ValueError(
            "ball_uv shape must be (T, 2), (N, T, 2), or (B, N, T, 2), "
            f"got {ball.shape}"
        )

    batch_size, num_cameras, num_frames = ball.shape[:3]
    vis_shape = (batch_size, num_cameras, num_frames)
    if ball_vis is None:
        vis = np.ones(vis_shape, dtype=np.float32)
    else:
        vis = np.asarray(ball_vis, dtype=np.float32)
        if vis.ndim == 1:
            vis = vis[None, None, ...]
        elif vis.ndim == 2:
            vis = vis[None, ...]
        elif vis.ndim != 3:
            raise ValueError(
                "ball_vis shape must be (T,), (N, T), or (B, N, T), "
                f"got {vis.shape}"
            )
        if vis.shape != vis_shape:
            raise ValueError(f"ball_vis shape must be {vis_shape}, got {vis.shape}")
        if not np.isfinite(vis).all():
            raise ValueError("ball_vis contains non-finite values")
    return ball.astype(np.float32), vis.astype(np.float32), vis_shape


def normalize_player_keypoint_sequence(
    human_kp_2d: NDArray[np.float32],
    human_kp_vis: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], tuple[int, int, int]]:
    """Normalize player keypoints to (P, N, T, 17, 2)."""
    human = _as_float32("human_kp_2d", human_kp_2d)
    if human.ndim == 4 and human.shape[-2:] == (17, 2):
        human = human[:, None, ...]
    elif human.ndim == 5 and human.shape[-2:] == (17, 2):
        human = human
    else:
        raise ValueError(
            "human_kp_2d shape must be (P, T, 17, 2) or (P, N, T, 17, 2), "
            f"got {human.shape}"
        )

    num_players, num_cameras, num_frames = human.shape[:3]
    vis_shape = (num_players, num_cameras, num_frames, 17)
    if human_kp_vis is None:
        vis = np.ones(vis_shape, dtype=np.float32)
    else:
        vis = np.asarray(human_kp_vis, dtype=np.float32)
        if vis.ndim == 3:
            vis = vis[:, None, ...]
        elif vis.ndim != 4:
            raise ValueError(
                "human_kp_vis shape must be (P, T, 17) or (P, N, T, 17), "
                f"got {vis.shape}"
            )
        if vis.shape != vis_shape:
            raise ValueError(f"human_kp_vis shape must be {vis_shape}, got {vis.shape}")
        if not np.isfinite(vis).all():
            raise ValueError("human_kp_vis contains non-finite values")
    return human.astype(np.float32), vis.astype(np.float32), (
        num_players,
        num_cameras,
        num_frames,
    )


def normalize_court_keypoint_sequence(
    court_kp: NDArray[np.float32],
    court_vis: NDArray[np.float32] | None,
    *,
    target_batch_size: int,
    target_num_cameras: int,
    target_num_frames: int,
    target_num_keypoints: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Normalize court keypoints to (B, N, T, K, 2)."""
    court = _as_float32("court_kp", court_kp)

    if court.ndim == 2 and court.shape[-1] == 2:
        court = np.broadcast_to(
            court[None, None, None, ...],
            (
                target_batch_size,
                target_num_cameras,
                target_num_frames,
                court.shape[-2],
                2,
            ),
        )
    elif court.ndim == 3 and court.shape[-1] == 2:
        if court.shape[0] == target_num_frames:
            court = np.broadcast_to(
                court[None, None, ...],
                (
                    target_batch_size,
                    target_num_cameras,
                    target_num_frames,
                    court.shape[-2],
                    2,
                ),
            )
        elif court.shape[0] == target_num_cameras:
            court = np.broadcast_to(
                court[None, :, None, ...],
                (
                    target_batch_size,
                    target_num_cameras,
                    target_num_frames,
                    court.shape[-2],
                    2,
                ),
            )
        else:
            raise ValueError(
                "court_kp shape (X, K, 2) must use X equal to T or N, "
                f"got X={court.shape[0]}, T={target_num_frames}, "
                f"N={target_num_cameras}"
            )
    elif court.ndim == 4 and court.shape[-1] == 2:
        if court.shape[:2] != (target_num_cameras, target_num_frames):
            raise ValueError(
                "court_kp shape (N, T, K, 2) must match target N/T, "
                f"got {court.shape[:2]}, expected "
                f"{(target_num_cameras, target_num_frames)}"
            )
        court = np.broadcast_to(
            court[None, ...],
            (target_batch_size, *court.shape),
        )
    elif court.ndim == 5 and court.shape[-1] == 2:
        if court.shape[:3] != (
            target_batch_size,
            target_num_cameras,
            target_num_frames,
        ):
            raise ValueError(
                "court_kp shape (B, N, T, K, 2) must match target B/N/T, "
                f"got {court.shape[:3]}, expected "
                f"{(target_batch_size, target_num_cameras, target_num_frames)}"
            )
    else:
        raise ValueError(
            "court_kp shape must be (K, 2), (T, K, 2), (N, K, 2), "
            "(N, T, K, 2), or (B, N, T, K, 2), "
            f"got {court.shape}"
        )

    vis_input_shape = court.shape[:-1]
    if court_vis is None:
        vis = np.ones(vis_input_shape, dtype=np.float32)
    else:
        raw_vis = np.asarray(court_vis, dtype=np.float32)
        if raw_vis.ndim == 1:
            vis = np.broadcast_to(
                raw_vis[None, None, None, ...],
                vis_input_shape,
            )
        elif raw_vis.ndim == 2:
            if raw_vis.shape[0] == target_num_frames:
                vis = np.broadcast_to(raw_vis[None, None, ...], vis_input_shape)
            elif raw_vis.shape[0] == target_num_cameras:
                vis = np.broadcast_to(raw_vis[None, :, None, ...], vis_input_shape)
            else:
                raise ValueError(
                    "court_vis shape (X, K) must use X equal to T or N, "
                    f"got X={raw_vis.shape[0]}, T={target_num_frames}, "
                    f"N={target_num_cameras}"
                )
        elif raw_vis.ndim == 3:
            if raw_vis.shape[:2] != (target_num_cameras, target_num_frames):
                raise ValueError(
                    "court_vis shape (N, T, K) must match target N/T, "
                    f"got {raw_vis.shape[:2]}, expected "
                    f"{(target_num_cameras, target_num_frames)}"
                )
            vis = np.broadcast_to(raw_vis[None, ...], vis_input_shape)
        elif raw_vis.ndim == 4:
            if raw_vis.shape != vis_input_shape:
                raise ValueError(
                    f"court_vis shape must be {vis_input_shape}, got {raw_vis.shape}"
                )
            vis = raw_vis
        else:
            raise ValueError(
                "court_vis shape must be (K,), (T, K), (N, K), "
                "(N, T, K), or (B, N, T, K), "
                f"got {raw_vis.shape}"
            )
        if not np.isfinite(vis).all():
            raise ValueError("court_vis contains non-finite values")

    return _pad_court_keypoints(
        np.array(court, dtype=np.float32, copy=True),
        np.array(vis, dtype=np.float32, copy=True),
        target_num_keypoints=target_num_keypoints,
    )
