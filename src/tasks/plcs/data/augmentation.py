"""PLCS-specific train-time augmentation for human and court keypoint inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.utils.data.augmentation import random_visibility_dropout

PLCSSample = dict[str, Tensor]


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


def _prob(config: Mapping[str, Any], *, default: float = 1.0) -> float:
    return float(config.get("prob", default))


def _should_apply(prob: float, reference: Tensor) -> bool:
    if prob <= 0:
        return False
    if prob >= 1:
        return True
    return bool(torch.rand((), device=reference.device).item() < prob)


def _clone_sample(sample: PLCSSample) -> PLCSSample:
    return {
        key: (value.clone() if isinstance(value, Tensor) else value)
        for key, value in sample.items()
    }


def _mask_keypoints(keypoints: Tensor, visibility: Tensor) -> Tensor:
    return keypoints * (visibility > 0).to(dtype=keypoints.dtype).unsqueeze(-1)


def _add_visible_gaussian_noise(
    keypoints: Tensor,
    visibility: Tensor,
    noise_std: float,
) -> Tensor:
    if noise_std <= 0:
        return keypoints
    visible = (visibility > 0).to(dtype=keypoints.dtype).unsqueeze(-1)
    noise = torch.randn_like(keypoints) * noise_std
    return (keypoints + noise * visible).clamp(0.0, 1.0)


class PLCSObservationAugmentation:
    """Apply configured observation corruption to PLCS keypoint inputs."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _as_dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.gaussian_cfg = self._gaussian_config()
        self.visibility_dropout_cfg = self._visibility_dropout_config()

    def _gaussian_config(self) -> dict[str, Any]:
        if "gaussian_noise" in self.config:
            return _as_dict(self.config.get("gaussian_noise"))
        noise_std = float(self.config.get("keypoint_noise_std", 0.0))
        return {
            "enabled": noise_std > 0,
            "prob": 1.0,
            "human_std": noise_std,
            "court_std": noise_std,
        }

    def _visibility_dropout_config(self) -> dict[str, Any]:
        if "visibility_dropout" in self.config:
            return _as_dict(self.config.get("visibility_dropout"))
        drop_prob = float(self.config.get("visibility_drop_prob", 0.0))
        return {
            "enabled": drop_prob > 0,
            "prob": 1.0,
            "human_drop_prob": drop_prob,
            "court_drop_prob": drop_prob,
        }

    def forward(self, sample: PLCSSample) -> PLCSSample:
        """Return an augmented PLCS sample."""
        if not self.enabled:
            return sample

        out = _clone_sample(sample)
        self._apply_gaussian_noise(out)
        self._apply_visibility_dropout(out)
        out["human_kp"] = _mask_keypoints(out["human_kp"], out["human_vis"])
        out["court_kp"] = _mask_keypoints(out["court_kp"], out["court_vis"])
        return out

    def _apply_gaussian_noise(self, sample: PLCSSample) -> None:
        cfg = self.gaussian_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), sample["human_kp"]):
            return
        human_std = float(cfg.get("human_std", cfg.get("keypoint_noise_std", 0.0)))
        court_std = float(cfg.get("court_std", human_std))
        sample["human_kp"] = _add_visible_gaussian_noise(
            sample["human_kp"],
            sample["human_vis"],
            human_std,
        )
        sample["court_kp"] = _add_visible_gaussian_noise(
            sample["court_kp"],
            sample["court_vis"],
            court_std,
        )

    def _apply_visibility_dropout(self, sample: PLCSSample) -> None:
        cfg = self.visibility_dropout_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), sample["human_vis"]):
            return
        sample["human_vis"] = random_visibility_dropout(
            sample["human_vis"],
            float(cfg.get("human_drop_prob", cfg.get("drop_prob", 0.0))),
        )
        sample["court_vis"] = random_visibility_dropout(
            sample["court_vis"],
            float(cfg.get("court_drop_prob", cfg.get("drop_prob", 0.0))),
        )