"""BLCS-specific composition of detector-inspired ball observation augmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.configuration import as_config_mapping, require_config_value
from src.tasks.base.data.augmentation import BaseObservationAugmentation
from src.tasks.blcs.data.types import BLCSMultiViewSample
from src.utils.configuration import (
    MissingConfigurationKeyError,
    UnknownConfigurationKeyError,
)
from src.utils.data.augmentation import (
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
from src.utils.tensor_utils import clone_tensor_dict


def _float_value(config: Mapping[str, object], key: str, *, path: str) -> float:
    value = require_config_value(config, key, (float, int), path=path)
    return float(cast("float | int", value))


def _int_value(config: Mapping[str, object], key: str, *, path: str) -> int:
    return cast("int", require_config_value(config, key, int, path=path))


class BLCSBallObservationAugmentation(BaseObservationAugmentation[BLCSMultiViewSample]):
    """Apply configured detector-like corruption to BLCS ball observations.

    The class intentionally modifies only observation tensors.  Clean 3D labels
    and camera parameters are left untouched, and optional clean 2D targets are
    preserved for reprojection-style losses before input corruption is applied.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        blocks = {
            "uv_scale": {
                "enabled",
                "prob",
                "scale_range",
                "apply_to_ball",
                "apply_to_court",
            },
            "gaussian_noise": {"enabled", "prob", "ball_std", "court_std"},
            "visibility_dropout": {"enabled", "prob", "drop_prob"},
            "temporal_jitter": {
                "enabled",
                "prob",
                "jitter_std",
                "drift_std",
                "drift_decay",
            },
            "burst_dropout": {
                "enabled",
                "prob",
                "track_prob",
                "min_len",
                "max_len",
                "max_bursts",
            },
            "false_positive": {
                "enabled",
                "prob",
                "prob_absent",
                "prob_after_dropout",
                "after_dropout_window",
            },
            "edge_degradation": {
                "enabled",
                "prob",
                "edge_margin",
                "noise_std",
                "drop_prob",
                "clip_out_prob",
            },
            "speed_conditioned": {
                "enabled",
                "prob",
                "frame_prob",
                "speed_threshold",
                "lag_overshoot_range",
                "noise_std",
            },
        }
        required = {"enabled", "preserve_clean_targets", *blocks}
        self._require_exact_keys(config, required, path="data.augmentation")
        for name, keys in blocks.items():
            child = as_config_mapping(config[name], path=f"data.augmentation.{name}")
            self._require_exact_keys(child, keys, path=f"data.augmentation.{name}")
        super().__init__(config)
        self.preserve_clean_targets = bool(self.config["preserve_clean_targets"])

    @staticmethod
    def _require_exact_keys(
        config: Mapping[str, Any], keys: set[str], *, path: str
    ) -> None:
        missing = sorted(keys - set(config))
        if missing:
            raise MissingConfigurationKeyError(
                f"Missing required configuration key(s): {', '.join(f'{path}.{key}' for key in missing)}."
            )
        unknown = sorted(set(config) - keys)
        if unknown:
            raise UnknownConfigurationKeyError(
                f"Unknown configuration key(s): {', '.join(f'{path}.{key}' for key in unknown)}."
            )

    def _uv_scale_config(self) -> dict[str, Any]:
        return dict(
            as_config_mapping(
                self.config["uv_scale"], path="data.augmentation.uv_scale"
            )
        )

    def _gaussian_config(self) -> dict[str, Any]:
        return dict(
            as_config_mapping(
                self.config["gaussian_noise"], path="data.augmentation.gaussian_noise"
            )
        )

    def _visibility_dropout_config(self) -> dict[str, Any]:
        return dict(
            as_config_mapping(
                self.config["visibility_dropout"],
                path="data.augmentation.visibility_dropout",
            )
        )

    def forward(self, sample: BLCSMultiViewSample) -> BLCSMultiViewSample:
        """Return an augmented BLCS sample."""
        if not self.enabled:
            return sample

        out = clone_tensor_dict(sample)
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
        if bool(cfg["apply_to_ball"]):
            sample["ball_uv"], sample["ball_vis"] = scale_uv_with_visibility(
                uv=sample["ball_uv"],
                visibility=sample["ball_vis"],
                scale=float(scale),
            )
        if bool(cfg["apply_to_court"]):
            sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
                uv=sample["court_kp"],
                visibility=sample["court_vis"],
                scale=float(scale),
            )

    def _apply_gaussian_noise(self, sample: BLCSMultiViewSample) -> None:
        cfg = self.gaussian_cfg
        if not self._active(cfg, sample["ball_uv"]):
            return
        ball_std = _float_value(
            cfg, "ball_std", path="data.augmentation.gaussian_noise"
        )
        court_std = _float_value(
            cfg, "court_std", path="data.augmentation.gaussian_noise"
        )
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
            jitter_std=_float_value(
                cfg, "jitter_std", path="data.augmentation.temporal_jitter"
            ),
            drift_std=_float_value(
                cfg, "drift_std", path="data.augmentation.temporal_jitter"
            ),
            drift_decay=_float_value(
                cfg, "drift_decay", path="data.augmentation.temporal_jitter"
            ),
        )

    def _apply_speed_conditioned(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.speed_conditioned_cfg
        if not self._active(cfg, ball_uv):
            return ball_uv, ball_vis
        lag_range = parse_float_range(
            cfg["lag_overshoot_range"],
            "data.augmentation.speed_conditioned.lag_overshoot_range",
        )
        return apply_speed_conditioned_localization_error(
            ball_uv,
            ball_vis,
            prob=_float_value(
                cfg, "frame_prob", path="data.augmentation.speed_conditioned"
            ),
            speed_threshold=_float_value(
                cfg, "speed_threshold", path="data.augmentation.speed_conditioned"
            ),
            lag_overshoot_range=lag_range,
            noise_std=_float_value(
                cfg, "noise_std", path="data.augmentation.speed_conditioned"
            ),
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
            edge_margin=_float_value(
                cfg, "edge_margin", path="data.augmentation.edge_degradation"
            ),
            noise_std=_float_value(
                cfg, "noise_std", path="data.augmentation.edge_degradation"
            ),
            drop_prob=_float_value(
                cfg, "drop_prob", path="data.augmentation.edge_degradation"
            ),
            clip_out_prob=_float_value(
                cfg, "clip_out_prob", path="data.augmentation.edge_degradation"
            ),
        )

    def _apply_visibility_dropout(self, ball_vis: Tensor) -> Tensor:
        cfg = self.visibility_dropout_cfg
        if not self._active(cfg, ball_vis):
            return ball_vis
        return random_visibility_dropout(
            ball_vis,
            _float_value(cfg, "drop_prob", path="data.augmentation.visibility_dropout"),
        )

    def _apply_burst_dropout(self, ball_vis: Tensor) -> Tensor:
        cfg = self.burst_dropout_cfg
        if not self._active(cfg, ball_vis):
            return ball_vis
        return apply_burst_visibility_dropout(
            ball_vis,
            prob=_float_value(
                cfg, "track_prob", path="data.augmentation.burst_dropout"
            ),
            min_len=_int_value(cfg, "min_len", path="data.augmentation.burst_dropout"),
            max_len=_int_value(cfg, "max_len", path="data.augmentation.burst_dropout"),
            max_bursts=_int_value(
                cfg, "max_bursts", path="data.augmentation.burst_dropout"
            ),
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
            false_positive_prob=_float_value(
                cfg, "prob_absent", path="data.augmentation.false_positive"
            ),
            after_dropout_mask=dropped_mask,
            after_dropout_prob=_float_value(
                cfg, "prob_after_dropout", path="data.augmentation.false_positive"
            ),
            after_dropout_window=_int_value(
                cfg, "after_dropout_window", path="data.augmentation.false_positive"
            ),
        )


__all__ = ["BLCSBallObservationAugmentation"]
