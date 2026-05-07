"""Event-detection-specific augmentation for UV observation inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from torch import Tensor

from src.tasks.blcs.data.augmentation import BLCSBallObservationAugmentation
from src.tasks.event_detection.data.types import EventUVSample


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _mask_points(points: Tensor, visibility: Tensor) -> Tensor:
    return points * (visibility > 0).to(dtype=points.dtype).unsqueeze(-1)


def _normalize_event_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = _as_dict(config)
    normalized = dict(cfg)
    normalized.setdefault("preserve_clean_targets", False)

    if "gaussian_noise" not in normalized and (
        "ball_noise_std" in cfg or "court_noise_std" in cfg
    ):
        normalized["gaussian_noise"] = {
            "enabled": True,
            "prob": 1.0,
            "ball_std": float(cfg.get("ball_noise_std", 0.0)),
            "court_std": float(cfg.get("court_noise_std", 0.0)),
        }

    if "visibility_dropout" not in normalized and (
        "ball_visibility_drop_prob" in cfg or "court_visibility_drop_prob" in cfg
    ):
        normalized["visibility_dropout"] = {
            "enabled": True,
            "prob": 1.0,
            "ball_drop_prob": float(cfg.get("ball_visibility_drop_prob", 0.0)),
            "court_drop_prob": float(cfg.get("court_visibility_drop_prob", 0.0)),
        }

    return normalized


class EventUVObservationAugmentation:
    """Apply configured corruption to UV event-detection inputs."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _normalize_event_config(config)
        self.augmentation = BLCSBallObservationAugmentation(self.config)

    def forward(self, sample: EventUVSample) -> EventUVSample:
        out = cast(EventUVSample, self.augmentation.forward(sample))
        out["ball_uv"] = _mask_points(out["ball_uv"], out["ball_vis"])
        out["court_kp"] = _mask_points(out["court_kp"], out["court_vis"])
        return out