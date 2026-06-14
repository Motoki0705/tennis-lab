"""BLCS-specific composition of detector-inspired ball observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.augmentation import BaseObservationAugmentation
from src.tasks.blcs.data.types import BLCSMultiViewSample
from src.utils.data.augmentation import (
    _as_dict,
    add_gaussian_noise,
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    apply_edge_aware_degradation,
    apply_speed_conditioned_localization_error,
    inject_false_positive_observations,
    parse_float_range,
    random_visibility_dropout,
    scale_uv_with_visibility,
)


def _clone_sample(sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
    return {
        key: (value.clone() if isinstance(value, Tensor) else value)
        for key, value in sample.items()
    }


class BLCSBallObservationAugmentation(BaseObservationAugmentation):
    """Apply configured detector-like corruption to BLCS ball observations.

    The class intentionally modifies only observation tensors.  Clean 3D labels
    and camera parameters are left untouched, and optional clean 2D targets are
    preserved for reprojection-style losses before input corruption is applied.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.preserve_clean_targets = bool(
            self.config.get("preserve_clean_targets", True)
        )

    def _uv_scale_config(self) -> dict[str, Any]:
        if "uv_scale" in self.config:
            return _as_dict(self.config.get("uv_scale"))
        scale_range = self.config.get("scale_range", [1.0, 1.0])
        scale_min, scale_max = parse_float_range(scale_range, "augmentation.scale_range")
        return {
            "enabled": not (scale_min == 1.0 and scale_max == 1.0),
            "prob": 1.0,
            "scale_range": [scale_min, scale_max],
            "apply_to_ball": True,
            "apply_to_court": True,
        }

    def _gaussian_config(self) -> dict[str, Any]:
        if "gaussian_noise" in self.config:
            return _as_dict(self.config.get("gaussian_noise"))
        uv_noise_std = float(self.config.get("uv_noise_std", 0.005))
        return {
            "enabled": uv_noise_std > 0,
            "prob": 1.0,
            "ball_std": uv_noise_std,
            "court_std": uv_noise_std,
        }

    def _visibility_dropout_config(self) -> dict[str, Any]:
        if "visibility_dropout" in self.config:
            return _as_dict(self.config.get("visibility_dropout"))
        drop_prob = float(self.config.get("visibility_drop_prob", 0.1))
        return {
            "enabled": drop_prob > 0,
            "prob": 1.0,
            "drop_prob": drop_prob,
        }

    def forward(self, sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
        """Return an augmented BLCS sample."""
        if not self.enabled:
            return sample

        out = _clone_sample(sample)
        ball_uv = out["ball_uv"]
        ball_vis = out["ball_vis"]
        if self.preserve_clean_targets:
            out["ball_uv_target"] = ball_uv.clone()
            out["ball_vis_target"] = ball_vis.clone()

        dropped_mask = torch.zeros_like(ball_vis, dtype=torch.bool)

        self._apply_uv_scale(out)
        self._apply_gaussian_noise(out)
        out["ball_uv"] = self._apply_temporal_jitter(out["ball_uv"], out["ball_vis"])

        before_vis = out["ball_vis"].clone()
        out["ball_uv"], out["ball_vis"] = self._apply_speed_conditioned(
            out["ball_uv"],
            out["ball_vis"],
        )
        dropped_mask |= (before_vis > 0) & (out["ball_vis"] <= 0)

        before_vis = out["ball_vis"].clone()
        out["ball_uv"], out["ball_vis"] = self._apply_edge_degradation(
            out["ball_uv"],
            out["ball_vis"],
        )
        dropped_mask |= (before_vis > 0) & (out["ball_vis"] <= 0)

        before_vis = out["ball_vis"].clone()
        out["ball_vis"] = self._apply_visibility_dropout(out["ball_vis"])
        dropped_mask |= (before_vis > 0) & (out["ball_vis"] <= 0)

        before_vis = out["ball_vis"].clone()
        out["ball_vis"] = self._apply_burst_dropout(out["ball_vis"])
        dropped_mask |= (before_vis > 0) & (out["ball_vis"] <= 0)

        out["ball_uv"], out["ball_vis"] = self._apply_false_positive(
            out["ball_uv"],
            out["ball_vis"],
            dropped_mask=dropped_mask,
        )

        out["ball_uv"] = out["ball_uv"].clamp(0.0, 1.0)
        out["court_kp"] = out["court_kp"].clamp(0.0, 1.0)
        return out

    def _apply_uv_scale(self, sample: BLCSMultiViewSample) -> None:
        cfg = self.uv_scale_cfg
        if not self._active(cfg, sample["ball_uv"]):
            return
        scale_min, scale_max = self._parse_scale_range(cfg)
        scale = (
            torch.rand((), device=sample["ball_uv"].device).item()
            * (scale_max - scale_min)
            + scale_min
        )
        if abs(scale - 1.0) < 1e-8:
            return
        if bool(cfg.get("apply_to_ball", True)):
            sample["ball_uv"], sample["ball_vis"] = scale_uv_with_visibility(
                uv=sample["ball_uv"],
                visibility=sample["ball_vis"],
                scale=float(scale),
            )
        if bool(cfg.get("apply_to_court", True)):
            sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
                uv=sample["court_kp"],
                visibility=sample["court_vis"],
                scale=float(scale),
            )

    def _apply_gaussian_noise(self, sample: BLCSMultiViewSample) -> None:
        cfg = self.gaussian_cfg
        if not self._active(cfg, sample["ball_uv"]):
            return
        ball_std = float(cfg.get("ball_std", cfg.get("uv_noise_std", 0.0)))
        court_std = float(cfg.get("court_std", ball_std))
        if ball_std > 0:
            sample["ball_uv"] = add_gaussian_noise(
                sample["ball_uv"],
                ball_std,
            ).clamp(0.0, 1.0)
        if court_std > 0:
            sample["court_kp"] = add_gaussian_noise(
                sample["court_kp"],
                court_std,
            ).clamp(0.0, 1.0)

    def _apply_temporal_jitter(self, ball_uv: Tensor, ball_vis: Tensor) -> Tensor:
        cfg = self.temporal_jitter_cfg
        if not self._active(cfg, ball_uv):
            return ball_uv
        return add_temporally_correlated_jitter(
            ball_uv,
            ball_vis,
            jitter_std=float(cfg.get("jitter_std", 0.0)),
            drift_std=float(cfg.get("drift_std", 0.0)),
            drift_decay=float(cfg.get("drift_decay", 0.9)),
        )

    def _apply_speed_conditioned(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.speed_conditioned_cfg
        if not self._active(cfg, ball_uv):
            return ball_uv, ball_vis
        return apply_speed_conditioned_localization_error(
            ball_uv,
            ball_vis,
            prob=float(cfg.get("frame_prob", 1.0)),
            speed_threshold=float(cfg.get("speed_threshold", 0.025)),
            lag_overshoot_range=cfg.get("lag_overshoot_range", [-0.2, 0.3]),
            noise_std=float(cfg.get("noise_std", 0.0)),
        )

    def _apply_edge_degradation(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.edge_degradation_cfg
        if not self._active(cfg, ball_uv):
            return ball_uv, ball_vis
        return apply_edge_aware_degradation(
            ball_uv,
            ball_vis,
            edge_margin=float(cfg.get("edge_margin", 0.08)),
            noise_std=float(cfg.get("noise_std", 0.0)),
            drop_prob=float(cfg.get("drop_prob", 0.0)),
            clip_out_prob=float(cfg.get("clip_out_prob", 0.0)),
        )

    def _apply_visibility_dropout(self, ball_vis: Tensor) -> Tensor:
        cfg = self.visibility_dropout_cfg
        if not self._active(cfg, ball_vis):
            return ball_vis
        return random_visibility_dropout(
            ball_vis,
            float(cfg.get("drop_prob", cfg.get("visibility_drop_prob", 0.0))),
        )

    def _apply_burst_dropout(self, ball_vis: Tensor) -> Tensor:
        cfg = self.burst_dropout_cfg
        if not self._active(cfg, ball_vis):
            return ball_vis
        return apply_burst_visibility_dropout(
            ball_vis,
            prob=float(cfg.get("track_prob", 1.0)),
            min_len=int(cfg.get("min_len", 2)),
            max_len=int(cfg.get("max_len", 6)),
            max_bursts=int(cfg.get("max_bursts", 1)),
        )

    def _apply_false_positive(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        *,
        dropped_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.false_positive_cfg
        if not self._active(cfg, ball_uv):
            return ball_uv, ball_vis
        return inject_false_positive_observations(
            ball_uv,
            ball_vis,
            false_positive_prob=float(cfg.get("prob_absent", 0.0)),
            after_dropout_mask=dropped_mask,
            after_dropout_prob=float(cfg.get("prob_after_dropout", 0.0)),
            after_dropout_window=int(cfg.get("after_dropout_window", 0)),
        )


__all__ = ["BLCSBallObservationAugmentation"]
