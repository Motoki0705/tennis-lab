"""PLCS-specific train-time augmentation for human and court keypoint inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from src.tasks.base.data.augmentation import BaseObservationAugmentation
from src.utils.data.augmentation import (
    _as_dict,
    _enabled,
    _prob,
    _should_apply,
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    apply_edge_aware_degradation,
    apply_speed_conditioned_localization_error,
    inject_false_positive_observations,
    parse_float_range,
    random_visibility_dropout,
    scale_uv_with_visibility,
)

PLCSSample = dict[str, Tensor]


def _clone_sample(sample: PLCSSample) -> PLCSSample:
    return {
        key: (value.clone() if isinstance(value, Tensor) else value)
        for key, value in sample.items()
    }


def _entity_value(
    config: Mapping[str, Any],
    entity: str,
    key: str,
    default: Any,
) -> Any:
    return config.get(f"{entity}_{key}", config.get(key, default))


def _flatten_temporal_tracks(
    keypoints: Tensor,
    visibility: Tensor,
) -> tuple[Tensor, Tensor, tuple[int, ...], int, int]:
    if keypoints.shape[:-1] != visibility.shape:
        raise ValueError(
            "visibility shape must match keypoints without coordinate dimension: "
            f"got visibility={tuple(visibility.shape)}, keypoints={tuple(keypoints.shape)}."
        )
    if keypoints.ndim < 3:
        raise ValueError(
            "keypoints must include temporal and track dimensions, got "
            f"{tuple(keypoints.shape)}."
        )

    prefix_shape = tuple(int(dim) for dim in keypoints.shape[:-3])
    time_len = int(keypoints.shape[-3])
    num_tracks = int(keypoints.shape[-2])
    flat_keypoints = keypoints.transpose(-3, -2).reshape(-1, time_len, 2)
    flat_visibility = visibility.transpose(-2, -1).reshape(-1, time_len)
    return flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks


def _restore_temporal_tracks(
    flat_keypoints: Tensor,
    flat_visibility: Tensor,
    *,
    prefix_shape: tuple[int, ...],
    time_len: int,
    num_tracks: int,
) -> tuple[Tensor, Tensor]:
    keypoints = flat_keypoints.reshape(*prefix_shape, num_tracks, time_len, 2)
    visibility = flat_visibility.reshape(*prefix_shape, num_tracks, time_len)
    return keypoints.transpose(-3, -2).contiguous(), visibility.transpose(-2, -1).contiguous()


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


class PLCSObservationAugmentation(BaseObservationAugmentation):
    """Apply configured observation corruption to PLCS keypoint inputs."""

    def _uv_scale_config(self) -> dict[str, Any]:
        if "uv_scale" in self.config:
            return _as_dict(self.config.get("uv_scale"))
        scale_range = self.config.get("scale_range", [1.0, 1.0])
        scale_min, scale_max = parse_float_range(scale_range, "augmentation.scale_range")
        return {
            "enabled": not (scale_min == 1.0 and scale_max == 1.0),
            "prob": 1.0,
            "scale_range": [scale_min, scale_max],
            "apply_to_human": True,
            "apply_to_court": True,
        }

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
        human_dropped_mask = torch.zeros_like(out["human_vis"], dtype=torch.bool)
        court_dropped_mask = torch.zeros_like(out["court_vis"], dtype=torch.bool)

        self._apply_uv_scale(out)
        self._apply_gaussian_noise(out)

        out["human_kp"] = self._apply_temporal_jitter(
            out["human_kp"],
            out["human_vis"],
            entity="human",
        )
        out["court_kp"] = self._apply_temporal_jitter(
            out["court_kp"],
            out["court_vis"],
            entity="court",
        )

        before_human_vis = out["human_vis"].clone()
        out["human_kp"], out["human_vis"] = self._apply_speed_conditioned(
            out["human_kp"],
            out["human_vis"],
            entity="human",
        )
        human_dropped_mask |= (before_human_vis > 0) & (out["human_vis"] <= 0)

        before_court_vis = out["court_vis"].clone()
        out["court_kp"], out["court_vis"] = self._apply_speed_conditioned(
            out["court_kp"],
            out["court_vis"],
            entity="court",
        )
        court_dropped_mask |= (before_court_vis > 0) & (out["court_vis"] <= 0)

        before_human_vis = out["human_vis"].clone()
        out["human_kp"], out["human_vis"] = self._apply_edge_degradation(
            out["human_kp"],
            out["human_vis"],
            entity="human",
        )
        human_dropped_mask |= (before_human_vis > 0) & (out["human_vis"] <= 0)

        before_court_vis = out["court_vis"].clone()
        out["court_kp"], out["court_vis"] = self._apply_edge_degradation(
            out["court_kp"],
            out["court_vis"],
            entity="court",
        )
        court_dropped_mask |= (before_court_vis > 0) & (out["court_vis"] <= 0)

        before_human_vis = out["human_vis"].clone()
        out["human_vis"] = self._apply_visibility_dropout(
            out["human_vis"],
            entity="human",
        )
        human_dropped_mask |= (before_human_vis > 0) & (out["human_vis"] <= 0)

        before_court_vis = out["court_vis"].clone()
        out["court_vis"] = self._apply_visibility_dropout(
            out["court_vis"],
            entity="court",
        )
        court_dropped_mask |= (before_court_vis > 0) & (out["court_vis"] <= 0)

        before_human_vis = out["human_vis"].clone()
        out["human_vis"] = self._apply_burst_dropout(
            out["human_vis"],
            entity="human",
        )
        human_dropped_mask |= (before_human_vis > 0) & (out["human_vis"] <= 0)

        before_court_vis = out["court_vis"].clone()
        out["court_vis"] = self._apply_burst_dropout(
            out["court_vis"],
            entity="court",
        )
        court_dropped_mask |= (before_court_vis > 0) & (out["court_vis"] <= 0)

        out["human_kp"], out["human_vis"] = self._apply_false_positive(
            out["human_kp"],
            out["human_vis"],
            entity="human",
            dropped_mask=human_dropped_mask,
        )
        out["court_kp"], out["court_vis"] = self._apply_false_positive(
            out["court_kp"],
            out["court_vis"],
            entity="court",
            dropped_mask=court_dropped_mask,
        )

        out["human_kp"] = _mask_keypoints(out["human_kp"], out["human_vis"])
        out["court_kp"] = _mask_keypoints(out["court_kp"], out["court_vis"])
        return out

    def _apply_uv_scale(self, sample: PLCSSample) -> None:
        cfg = self.uv_scale_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), sample["human_kp"]):
            return
        scale_min, scale_max = parse_float_range(
            cfg.get("scale_range", [1.0, 1.0]),
            "augmentation.uv_scale.scale_range",
        )
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError("augmentation.uv_scale.scale_range values must be positive.")
        scale = (
            torch.rand((), device=sample["human_kp"].device).item()
            * (scale_max - scale_min)
            + scale_min
        )
        if abs(scale - 1.0) < 1e-8:
            return
        if bool(cfg.get("apply_to_human", True)):
            sample["human_kp"], sample["human_vis"] = scale_uv_with_visibility(
                sample["human_kp"],
                sample["human_vis"],
                float(scale),
            )
        if bool(cfg.get("apply_to_court", True)):
            sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
                sample["court_kp"],
                sample["court_vis"],
                float(scale),
            )

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

    def _apply_temporal_jitter(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        entity: str,
    ) -> Tensor:
        cfg = self.temporal_jitter_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), keypoints):
            return keypoints
        jitter_std = float(_entity_value(cfg, entity, "jitter_std", 0.0))
        drift_std = float(_entity_value(cfg, entity, "drift_std", 0.0))
        if jitter_std <= 0 and drift_std <= 0:
            return keypoints
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = _flatten_temporal_tracks(
            keypoints,
            visibility,
        )
        flat_keypoints = add_temporally_correlated_jitter(
            flat_keypoints,
            flat_visibility,
            jitter_std=jitter_std,
            drift_std=drift_std,
            drift_decay=float(cfg.get("drift_decay", 0.9)),
        )
        restored_keypoints, _ = _restore_temporal_tracks(
            flat_keypoints,
            flat_visibility,
            prefix_shape=prefix_shape,
            time_len=time_len,
            num_tracks=num_tracks,
        )
        return restored_keypoints

    def _apply_speed_conditioned(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        entity: str,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.speed_conditioned_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), keypoints):
            return keypoints, visibility
        frame_prob = float(_entity_value(cfg, entity, "frame_prob", 0.0))
        if frame_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = _flatten_temporal_tracks(
            keypoints,
            visibility,
        )
        flat_keypoints, flat_visibility = apply_speed_conditioned_localization_error(
            flat_keypoints,
            flat_visibility,
            prob=frame_prob,
            speed_threshold=float(_entity_value(cfg, entity, "speed_threshold", 0.025)),
            lag_overshoot_range=_entity_value(
                cfg,
                entity,
                "lag_overshoot_range",
                [-0.2, 0.3],
            ),
            noise_std=float(_entity_value(cfg, entity, "noise_std", 0.0)),
        )
        return _restore_temporal_tracks(
            flat_keypoints,
            flat_visibility,
            prefix_shape=prefix_shape,
            time_len=time_len,
            num_tracks=num_tracks,
        )

    def _apply_edge_degradation(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        entity: str,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.edge_degradation_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), keypoints):
            return keypoints, visibility
        noise_std = float(_entity_value(cfg, entity, "noise_std", 0.0))
        drop_prob = float(_entity_value(cfg, entity, "drop_prob", 0.0))
        clip_out_prob = float(_entity_value(cfg, entity, "clip_out_prob", 0.0))
        if noise_std <= 0 and drop_prob <= 0 and clip_out_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = _flatten_temporal_tracks(
            keypoints,
            visibility,
        )
        flat_keypoints, flat_visibility = apply_edge_aware_degradation(
            flat_keypoints,
            flat_visibility,
            edge_margin=float(cfg.get("edge_margin", 0.08)),
            noise_std=noise_std,
            drop_prob=drop_prob,
            clip_out_prob=clip_out_prob,
        )
        return _restore_temporal_tracks(
            flat_keypoints,
            flat_visibility,
            prefix_shape=prefix_shape,
            time_len=time_len,
            num_tracks=num_tracks,
        )

    def _apply_visibility_dropout(
        self,
        visibility: Tensor,
        *,
        entity: str,
    ) -> Tensor:
        cfg = self.visibility_dropout_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), visibility):
            return visibility
        drop_prob = float(_entity_value(cfg, entity, "drop_prob", 0.0))
        return random_visibility_dropout(visibility, drop_prob)

    def _apply_burst_dropout(
        self,
        visibility: Tensor,
        *,
        entity: str,
    ) -> Tensor:
        cfg = self.burst_dropout_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), visibility):
            return visibility
        track_prob = float(_entity_value(cfg, entity, "track_prob", 0.0))
        if track_prob <= 0:
            return visibility
        flat_visibility = visibility.transpose(-2, -1).reshape(-1, visibility.shape[-2])
        flat_visibility = apply_burst_visibility_dropout(
            flat_visibility,
            prob=track_prob,
            min_len=int(cfg.get("min_len", 2)),
            max_len=int(cfg.get("max_len", 6)),
            max_bursts=int(cfg.get("max_bursts", 1)),
        )
        return flat_visibility.reshape(*visibility.shape[:-2], visibility.shape[-1], visibility.shape[-2]).transpose(-2, -1).contiguous()

    def _apply_false_positive(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        entity: str,
        dropped_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.false_positive_cfg
        if not _enabled(cfg) or not _should_apply(_prob(cfg), keypoints):
            return keypoints, visibility
        absent_prob = float(_entity_value(cfg, entity, "prob_absent", 0.0))
        after_dropout_prob = float(_entity_value(cfg, entity, "prob_after_dropout", 0.0))
        if absent_prob <= 0 and after_dropout_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = _flatten_temporal_tracks(
            keypoints,
            visibility,
        )
        _, flat_dropped_mask, _, _, _ = _flatten_temporal_tracks(
            keypoints,
            dropped_mask.to(dtype=visibility.dtype),
        )
        flat_keypoints, flat_visibility = inject_false_positive_observations(
            flat_keypoints,
            flat_visibility,
            false_positive_prob=absent_prob,
            after_dropout_mask=flat_dropped_mask > 0,
            after_dropout_prob=after_dropout_prob,
            after_dropout_window=int(_entity_value(cfg, entity, "after_dropout_window", 0)),
        )
        return _restore_temporal_tracks(
            flat_keypoints,
            flat_visibility,
            prefix_shape=prefix_shape,
            time_len=time_len,
            num_tracks=num_tracks,
        )