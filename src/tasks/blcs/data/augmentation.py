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
    SemanticConfigurationError,
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
        self._uv_scale_activation = self._activation(
            self.uv_scale_cfg, path="data.augmentation.uv_scale"
        )
        self._gaussian_activation = self._activation(
            self.gaussian_cfg, path="data.augmentation.gaussian_noise"
        )
        self._visibility_dropout_activation = self._activation(
            self.visibility_dropout_cfg,
            path="data.augmentation.visibility_dropout",
        )
        self._temporal_jitter_activation = self._activation(
            self.temporal_jitter_cfg, path="data.augmentation.temporal_jitter"
        )
        self._burst_dropout_activation = self._activation(
            self.burst_dropout_cfg, path="data.augmentation.burst_dropout"
        )
        self._false_positive_activation = self._activation(
            self.false_positive_cfg, path="data.augmentation.false_positive"
        )
        self._edge_degradation_activation = self._activation(
            self.edge_degradation_cfg, path="data.augmentation.edge_degradation"
        )
        self._speed_conditioned_activation = self._activation(
            self.speed_conditioned_cfg, path="data.augmentation.speed_conditioned"
        )

        self._uv_scale_range = self._parse_scale_range(self.uv_scale_cfg)
        self._uv_ball_mix = float(bool(self.uv_scale_cfg["apply_to_ball"]))
        self._uv_court_mix = float(bool(self.uv_scale_cfg["apply_to_court"]))
        self._gaussian_ball_std = _float_value(
            self.gaussian_cfg,
            "ball_std",
            path="data.augmentation.gaussian_noise",
        )
        self._gaussian_court_std = _float_value(
            self.gaussian_cfg,
            "court_std",
            path="data.augmentation.gaussian_noise",
        )
        self._jitter_std = _float_value(
            self.temporal_jitter_cfg,
            "jitter_std",
            path="data.augmentation.temporal_jitter",
        )
        self._drift_std = _float_value(
            self.temporal_jitter_cfg,
            "drift_std",
            path="data.augmentation.temporal_jitter",
        )
        self._drift_decay = _float_value(
            self.temporal_jitter_cfg,
            "drift_decay",
            path="data.augmentation.temporal_jitter",
        )
        self._speed_frame_prob = _float_value(
            self.speed_conditioned_cfg,
            "frame_prob",
            path="data.augmentation.speed_conditioned",
        )
        self._speed_threshold = _float_value(
            self.speed_conditioned_cfg,
            "speed_threshold",
            path="data.augmentation.speed_conditioned",
        )
        self._speed_lag_range = parse_float_range(
            self.speed_conditioned_cfg["lag_overshoot_range"],
            "data.augmentation.speed_conditioned.lag_overshoot_range",
        )
        self._speed_noise_std = _float_value(
            self.speed_conditioned_cfg,
            "noise_std",
            path="data.augmentation.speed_conditioned",
        )
        self._edge_margin = _float_value(
            self.edge_degradation_cfg,
            "edge_margin",
            path="data.augmentation.edge_degradation",
        )
        self._edge_noise_std = _float_value(
            self.edge_degradation_cfg,
            "noise_std",
            path="data.augmentation.edge_degradation",
        )
        self._edge_drop_prob = _float_value(
            self.edge_degradation_cfg,
            "drop_prob",
            path="data.augmentation.edge_degradation",
        )
        self._edge_clip_out_prob = _float_value(
            self.edge_degradation_cfg,
            "clip_out_prob",
            path="data.augmentation.edge_degradation",
        )
        self._visibility_drop_prob = _float_value(
            self.visibility_dropout_cfg,
            "drop_prob",
            path="data.augmentation.visibility_dropout",
        )
        self._burst_track_prob = _float_value(
            self.burst_dropout_cfg,
            "track_prob",
            path="data.augmentation.burst_dropout",
        )
        self._burst_min_len = _int_value(
            self.burst_dropout_cfg,
            "min_len",
            path="data.augmentation.burst_dropout",
        )
        self._burst_max_len = _int_value(
            self.burst_dropout_cfg,
            "max_len",
            path="data.augmentation.burst_dropout",
        )
        self._burst_max_bursts = _int_value(
            self.burst_dropout_cfg,
            "max_bursts",
            path="data.augmentation.burst_dropout",
        )
        self._false_positive_prob = _float_value(
            self.false_positive_cfg,
            "prob_absent",
            path="data.augmentation.false_positive",
        )
        self._after_dropout_prob = _float_value(
            self.false_positive_cfg,
            "prob_after_dropout",
            path="data.augmentation.false_positive",
        )
        self._after_dropout_window = _int_value(
            self.false_positive_cfg,
            "after_dropout_window",
            path="data.augmentation.false_positive",
        )

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

    @staticmethod
    def _activation(config: Mapping[str, object], *, path: str) -> tuple[bool, float]:
        enabled = cast("bool", require_config_value(config, "enabled", bool, path=path))
        probability = _float_value(config, "prob", path=path)
        if not 0.0 <= probability <= 1.0:
            raise SemanticConfigurationError(
                f"{path}.prob must be within [0, 1]; got {probability}."
            )
        return enabled, probability

    @staticmethod
    def _sample_activation(activation: tuple[bool, float], reference: Tensor) -> bool:
        enabled, probability = activation
        if not enabled or probability == 0.0:
            return False
        if probability == 1.0:
            return True
        return bool(torch.rand((), device=reference.device).item() < probability)

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
        if not self._sample_activation(self._uv_scale_activation, sample["ball_uv"]):
            return
        scale_min, scale_max = self._uv_scale_range
        scale = (
            torch.rand((), device=sample["ball_uv"].device).item()
            * (scale_max - scale_min)
            + scale_min
        )
        ball_scale = 1.0 + self._uv_ball_mix * (scale - 1.0)
        court_scale = 1.0 + self._uv_court_mix * (scale - 1.0)
        sample["ball_uv"], sample["ball_vis"] = scale_uv_with_visibility(
            uv=sample["ball_uv"],
            visibility=sample["ball_vis"],
            scale=ball_scale,
        )
        sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
            uv=sample["court_kp"],
            visibility=sample["court_vis"],
            scale=court_scale,
        )

    def _apply_gaussian_noise(self, sample: BLCSMultiViewSample) -> None:
        if not self._sample_activation(self._gaussian_activation, sample["ball_uv"]):
            return
        if self._gaussian_ball_std > 0:
            sample["ball_uv"] = add_gaussian_noise(
                sample["ball_uv"],
                self._gaussian_ball_std,
            ).clamp(0.0, 1.0)
        if self._gaussian_court_std > 0:
            sample["court_kp"] = add_gaussian_noise(
                sample["court_kp"],
                self._gaussian_court_std,
            ).clamp(0.0, 1.0)

    def _apply_temporal_jitter(self, ball_uv: Tensor, ball_vis: Tensor) -> Tensor:
        if not self._sample_activation(self._temporal_jitter_activation, ball_uv):
            return ball_uv
        return add_temporally_correlated_jitter(
            ball_uv,
            ball_vis,
            jitter_std=self._jitter_std,
            drift_std=self._drift_std,
            drift_decay=self._drift_decay,
        )

    def _apply_speed_conditioned(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self._sample_activation(self._speed_conditioned_activation, ball_uv):
            return ball_uv, ball_vis
        result: tuple[Tensor, Tensor] = apply_speed_conditioned_localization_error(
            ball_uv,
            ball_vis,
            prob=self._speed_frame_prob,
            speed_threshold=self._speed_threshold,
            lag_overshoot_range=self._speed_lag_range,
            noise_std=self._speed_noise_std,
        )
        return result

    def _apply_edge_degradation(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self._sample_activation(self._edge_degradation_activation, ball_uv):
            return ball_uv, ball_vis
        result: tuple[Tensor, Tensor] = apply_edge_aware_degradation(
            ball_uv,
            ball_vis,
            edge_margin=self._edge_margin,
            noise_std=self._edge_noise_std,
            drop_prob=self._edge_drop_prob,
            clip_out_prob=self._edge_clip_out_prob,
        )
        return result

    def _apply_visibility_dropout(self, ball_vis: Tensor) -> Tensor:
        if not self._sample_activation(self._visibility_dropout_activation, ball_vis):
            return ball_vis
        return random_visibility_dropout(ball_vis, self._visibility_drop_prob)

    def _apply_burst_dropout(self, ball_vis: Tensor) -> Tensor:
        if not self._sample_activation(self._burst_dropout_activation, ball_vis):
            return ball_vis
        return apply_burst_visibility_dropout(
            ball_vis,
            prob=self._burst_track_prob,
            min_len=self._burst_min_len,
            max_len=self._burst_max_len,
            max_bursts=self._burst_max_bursts,
        )

    def _apply_false_positive(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        *,
        dropped_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self._sample_activation(self._false_positive_activation, ball_uv):
            return ball_uv, ball_vis
        result: tuple[Tensor, Tensor] = inject_false_positive_observations(
            ball_uv,
            ball_vis,
            false_positive_prob=self._false_positive_prob,
            after_dropout_mask=dropped_mask,
            after_dropout_prob=self._after_dropout_prob,
            after_dropout_window=self._after_dropout_window,
        )
        return result


__all__ = ["BLCSBallObservationAugmentation"]
