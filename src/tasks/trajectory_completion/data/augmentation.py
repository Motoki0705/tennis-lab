"""Trajectory-completion-specific train-time augmentation orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from src.tasks.trajectory_completion.data.argument import TrajectoryArgumenter
from src.tasks.trajectory_completion.data.types import TrajectoryCompletionSample


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert plain dicts or DictConfig-like objects into a shallow dict."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _enabled(config: Mapping[str, Any], *, default: bool = False) -> bool:
    return bool(config.get("enabled", default))


def _parse_ratio(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must be a two-element list/tuple.")
    return int(value[0]), int(value[1])


def resolve_trajectory_argument_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve nested augmentation config into TrajectoryArgumenter kwargs."""
    cfg = _as_dict(config)
    visibility_cfg = _as_dict(cfg.get("visibility_dropout"))
    event_cfg = _as_dict(cfg.get("event_dropout"))
    gaussian_cfg = _as_dict(cfg.get("gaussian_noise"))
    outlier_cfg = _as_dict(cfg.get("outlier"))

    if visibility_cfg:
        point_dropout_prob = (
            float(visibility_cfg.get("drop_prob", 0.05))
            if _enabled(visibility_cfg, default=True)
            else 0.0
        )
    else:
        point_dropout_prob = float(cfg.get("point_dropout_prob", 0.05))

    if event_cfg:
        event_dropout_prob = (
            float(event_cfg.get("drop_prob", 0.0))
            if _enabled(event_cfg, default=True)
            else 0.0
        )
        event_window = int(event_cfg.get("event_window", 2))
        event_ratio = _parse_ratio(
            event_cfg.get("event_ratio", [2, 1]),
            "augmentation.event_dropout.event_ratio",
        )
        event_center_std = event_cfg.get("event_center_std")
    else:
        event_dropout_prob = float(cfg.get("event_dropout_prob", 0.0))
        event_window = int(cfg.get("event_window", 2))
        event_ratio = _parse_ratio(
            cfg.get("event_ratio", [2, 1]),
            "data.argument.event_ratio",
        )
        event_center_std = cfg.get("event_center_std")

    if gaussian_cfg:
        noise_std = (
            float(gaussian_cfg.get("noise_std", 0.01))
            if _enabled(gaussian_cfg, default=True)
            else 0.0
        )
        clamp_unit = bool(gaussian_cfg.get("clamp_unit", True))
    else:
        noise_std = float(cfg.get("noise_std", 0.01))
        clamp_unit = bool(cfg.get("clamp_unit", True))

    if outlier_cfg:
        outlier_prob = (
            float(outlier_cfg.get("outlier_prob", 0.0))
            if _enabled(outlier_cfg, default=True)
            else 0.0
        )
    else:
        outlier_prob = float(cfg.get("outlier_prob", 0.0))

    return {
        "point_dropout_prob": point_dropout_prob,
        "event_dropout_prob": event_dropout_prob,
        "event_window": event_window,
        "event_ratio": event_ratio,
        "event_center_std": event_center_std,
        "noise_std": noise_std,
        "clamp_unit": clamp_unit,
        "outlier_prob": outlier_prob,
    }


class TrajectoryObservationAugmentation:
    """Apply configured input corruption to trajectory completion samples."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _as_dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.argument_cfg = resolve_trajectory_argument_config(self.config)
        self.argumenter = TrajectoryArgumenter(self.argument_cfg)

    def forward(
        self,
        sample: TrajectoryCompletionSample,
        *,
        event_frames: Mapping[str, Tensor] | None = None,
    ) -> TrajectoryCompletionSample:
        """Return an augmented trajectory-completion sample."""
        if not self.enabled:
            return sample

        out: TrajectoryCompletionSample = {
            key: (value.clone() if isinstance(value, Tensor) else value)
            for key, value in sample.items()
        }
        ball_uv, ball_vis = self.argumenter(
            out["ball_uv_gt"],
            out["ball_gt_vis"],
            event_frames=event_frames,
        )
        out["ball_uv"] = ball_uv
        out["ball_vis"] = ball_vis
        return out