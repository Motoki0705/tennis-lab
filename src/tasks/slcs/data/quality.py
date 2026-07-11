"""Pseudo-label quality filtering for SLCS training targets.

The tennis_scene pseudo-annotations are model outputs, not ground truth. This
module converts observation coverage into explicit per-frame label validity
masks and confidence weights, so the loss never treats a hallucinated label as
a fully trusted one. Rationale (documented in the task README):

- **Player labels** (position/yaw) come from PLCS + GVHMR, whose reliability
  tracks 2D pose coverage. The per-frame confidence is the mean 2D keypoint
  visibility over joints and cameras; frames below ``min_player_confidence``
  are masked out entirely, the rest are weighted by
  ``confidence ** label_weight_power``.
- **Ball labels** come from BLCS, whose reliability tracks how many cameras
  observed the ball. Frames observed by fewer than ``min_ball_cameras``
  cameras are masked out; the weight is the observing-camera fraction raised
  to ``label_weight_power``.
- **Windows** whose labeled fraction falls below ``min_window_label_ratio``
  are dropped at dataset construction time (counted, never silent).

All functions are pure NumPy; non-finite labels are always invalid regardless
of coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds controlling pseudo-label filtering."""

    min_player_confidence: float = 0.3
    min_ball_cameras: int = 1
    label_weight_power: float = 1.0
    min_window_label_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_player_confidence <= 1.0:
            raise ValueError(
                f"min_player_confidence must be in [0, 1], got {self.min_player_confidence}."
            )
        if self.min_ball_cameras < 1:
            raise ValueError(f"min_ball_cameras must be >= 1, got {self.min_ball_cameras}.")
        if self.label_weight_power < 0.0:
            raise ValueError(
                f"label_weight_power must be >= 0, got {self.label_weight_power}."
            )
        if not 0.0 <= self.min_window_label_ratio <= 1.0:
            raise ValueError(
                f"min_window_label_ratio must be in [0, 1], got {self.min_window_label_ratio}."
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> QualityConfig:
        return cls(
            min_player_confidence=float(cfg.get("min_player_confidence", 0.3)),
            min_ball_cameras=int(cfg.get("min_ball_cameras", 1)),
            label_weight_power=float(cfg.get("label_weight_power", 1.0)),
            min_window_label_ratio=float(cfg.get("min_window_label_ratio", 0.1)),
        )


def player_label_confidence(human_kp_vis: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-player, per-frame label confidence from 2D pose coverage.

    Args:
        human_kp_vis: ``(P, N, T, J)`` keypoint visibility in ``[0, 1]``.

    Returns:
        ``(P, T)`` mean visibility over cameras and joints.
    """
    if human_kp_vis.ndim != 4:
        raise ValueError(f"human_kp_vis must be (P, N, T, J), got shape {human_kp_vis.shape}.")
    return np.asarray(
        human_kp_vis.astype(np.float32).mean(axis=(1, 3)), dtype=np.float32
    )


def ball_label_confidence(ball_vis: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Per-frame ball label confidence: fraction of cameras observing the ball.

    Args:
        ball_vis: ``(N, T)`` per-camera ball visibility.

    Returns:
        ``(T,)`` observing-camera fraction in ``[0, 1]``.
    """
    if ball_vis.ndim != 2:
        raise ValueError(f"ball_vis must be (N, T), got shape {ball_vis.shape}.")
    return np.asarray(ball_vis.astype(np.float32).mean(axis=0), dtype=np.float32)


def build_label_masks(
    *,
    human_kp_vis: NDArray[np.float32],
    ball_vis: NDArray[np.bool_],
    player_position: NDArray[np.float32],
    player_yaw: NDArray[np.float32],
    ball_3d: NDArray[np.float32],
    config: QualityConfig,
) -> dict[str, NDArray[Any]]:
    """Compute label validity masks and confidence weights for a whole clip.

    Returns a dict with:
        - ``player_label_valid``: ``(P, T)`` bool
        - ``player_label_weight``: ``(P, T)`` float32, zero where invalid
        - ``ball_label_valid``: ``(T,)`` bool
        - ``ball_label_weight``: ``(T,)`` float32, zero where invalid
    """
    num_cameras = ball_vis.shape[0]
    player_conf = player_label_confidence(human_kp_vis)
    player_finite = (
        np.isfinite(player_position).all(axis=-1) & np.isfinite(player_yaw)
    )
    if player_conf.shape != player_finite.shape:
        raise ValueError(
            f"pose coverage shape {player_conf.shape} does not match player label "
            f"shape {player_finite.shape}."
        )
    player_valid = player_finite & (player_conf >= config.min_player_confidence)
    player_weight = np.where(
        player_valid, player_conf**config.label_weight_power, 0.0
    ).astype(np.float32)

    ball_conf = ball_label_confidence(ball_vis)
    ball_finite = np.isfinite(ball_3d).all(axis=-1)
    if ball_conf.shape != ball_finite.shape:
        raise ValueError(
            f"ball coverage shape {ball_conf.shape} does not match ball label "
            f"shape {ball_finite.shape}."
        )
    min_fraction = config.min_ball_cameras / num_cameras
    if config.min_ball_cameras > num_cameras:
        raise ValueError(
            f"min_ball_cameras={config.min_ball_cameras} exceeds the clip's camera "
            f"count {num_cameras}; lower the threshold or exclude the clip explicitly."
        )
    ball_valid = ball_finite & (ball_conf >= min_fraction - 1e-9)
    ball_weight = np.where(ball_valid, ball_conf**config.label_weight_power, 0.0).astype(
        np.float32
    )

    return {
        "player_label_valid": player_valid,
        "player_label_weight": player_weight,
        "ball_label_valid": ball_valid,
        "ball_label_weight": ball_weight,
    }


def window_label_ratio(
    player_label_valid: NDArray[np.bool_],
    ball_label_valid: NDArray[np.bool_],
    *,
    start: int,
    length: int,
) -> float:
    """Fraction of window frames carrying at least one valid label."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}.")
    player_any = player_label_valid[:, start : start + length].any(axis=0)
    ball_any = ball_label_valid[start : start + length]
    return float((player_any | ball_any).mean())


__all__ = [
    "QualityConfig",
    "ball_label_confidence",
    "build_label_masks",
    "player_label_confidence",
    "window_label_ratio",
]
