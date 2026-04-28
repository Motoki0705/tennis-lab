"""Event-detection-specific augmentation for UV observation inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.event_detection.data.types import EventUVSample
from src.utils.data.augmentation import random_visibility_dropout


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return {}


def _enabled(config: Mapping[str, Any], *, default: bool = False) -> bool:
    return bool(config.get("enabled", default))


def _prob(config: Mapping[str, Any], *, default: float = 1.0) -> float:
    return float(config.get("prob", default))


def _should_apply(prob: float, reference: Tensor) -> bool:
    if prob <= 0:
        return False
    if prob >= 1:
        return True
    return bool(torch.rand((), device=reference.device).item() < prob)


def _clone_sample(sample: EventUVSample) -> EventUVSample:
    return {
        key: (value.clone() if isinstance(value, Tensor) else value)
        for key, value in sample.items()
    }


def _mask_points(points: Tensor, visibility: Tensor) -> Tensor:
    return points * (visibility > 0).to(dtype=points.dtype).unsqueeze(-1)


def _add_visible_gaussian_noise(points: Tensor, visibility: Tensor, noise_std: float) -> Tensor:
    if noise_std <= 0:
        return points
    visible = (visibility > 0).to(dtype=points.dtype).unsqueeze(-1)
    noise = torch.randn_like(points) * noise_std
    return (points + noise * visible).clamp(0.0, 1.0)


class EventUVObservationAugmentation:
    """Apply configured corruption to UV event-detection inputs."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _as_dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.gaussian_cfg = self._gaussian_config()
        self.visibility_dropout_cfg = self._visibility_dropout_config()

    def _gaussian_config(self) -> dict[str, Any]:
        if "gaussian_noise" in self.config:
            return _as_dict(self.config.get("gaussian_noise"))
        ball_std = float(self.config.get("ball_noise_std", 0.0))
        court_std = float(self.config.get("court_noise_std", ball_std))
        return {
            "enabled": ball_std > 0 or court_std > 0,
            "prob": 1.0,
            "ball_std": ball_std,
            "court_std": court_std,
        }

    def _visibility_dropout_config(self) -> dict[str, Any]:
        if "visibility_dropout" in self.config:
            return _as_dict(self.config.get("visibility_dropout"))
        ball_drop_prob = float(self.config.get("ball_visibility_drop_prob", 0.0))
        court_drop_prob = float(
            self.config.get("court_visibility_drop_prob", 0.0)
        )
        return {
            "enabled": ball_drop_prob > 0 or court_drop_prob > 0,
            "prob": 1.0,
            "ball_drop_prob": ball_drop_prob,
            "court_drop_prob": court_drop_prob,
        }

    def forward(self, sample: EventUVSample) -> EventUVSample:
        if not self.enabled:
            return sample

        out = _clone_sample(sample)
        self._apply_gaussian_noise(out)
        self._apply_visibility_dropout(out)
        out["ball_uv"] = _mask_points(out["ball_uv"], out["ball_vis"])
        out["court_kp"] = _mask_points(out["court_kp"], out["court_vis"])
        return out

    def _apply_gaussian_noise(self, sample: EventUVSample) -> None:
        cfg = self.gaussian_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), sample["ball_uv"]):
            return
        ball_std = float(cfg.get("ball_std", cfg.get("noise_std", 0.0)))
        court_std = float(cfg.get("court_std", ball_std))
        sample["ball_uv"] = _add_visible_gaussian_noise(
            sample["ball_uv"],
            sample["ball_vis"],
            ball_std,
        )
        sample["court_kp"] = _add_visible_gaussian_noise(
            sample["court_kp"],
            sample["court_vis"],
            court_std,
        )

    def _apply_visibility_dropout(self, sample: EventUVSample) -> None:
        cfg = self.visibility_dropout_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), sample["ball_vis"]):
            return
        sample["ball_vis"] = random_visibility_dropout(
            sample["ball_vis"],
            float(cfg.get("ball_drop_prob", cfg.get("drop_prob", 0.0))),
        )
        sample["court_vis"] = random_visibility_dropout(
            sample["court_vis"],
            float(cfg.get("court_drop_prob", 0.0)),
        )