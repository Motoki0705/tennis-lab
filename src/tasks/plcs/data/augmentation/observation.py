"""PLCS-specific train-time augmentation for human and court keypoint inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, cast

import torch
from torch import Tensor

from src.tasks.base.data.augmentation import BaseObservationAugmentation
from src.tasks.plcs.configuration import validate_augmentation
from src.utils.data.augmentation import (
    add_temporally_correlated_jitter,
    apply_burst_visibility_dropout,
    apply_edge_aware_degradation,
    apply_speed_conditioned_localization_error,
    inject_false_positive_observations,
    random_visibility_dropout,
    scale_uv_with_visibility,
)
from src.utils.tensor_utils import clone_tensor_dict

PLCSSample = dict[str, Tensor]


class PLCSObservationTrackingResult(NamedTuple):
    """Augmented sample plus visibility immediately before human FP injection."""

    sample: PLCSSample
    human_visibility_before_false_positive: Tensor


def _float_object(value: object) -> float:
    return float(cast("float | int", value))


def _int_object(value: object) -> int:
    return cast("int", value)


def _entity_setting(
    config: Mapping[str, Any],
    entity: Literal["human", "court"],
    key: str,
) -> Any:
    return config[f"{entity}_{key}"]


def _flatten_temporal_tracks(
    keypoints: Tensor,
    visibility: Tensor,
) -> tuple[Tensor, Tensor, tuple[int, ...], int, int]:
    prefix_shape = tuple(keypoints.shape[:-3])
    time_len = keypoints.shape[-3]
    num_tracks = keypoints.shape[-2]
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
    return keypoints.transpose(-3, -2).contiguous(), visibility.transpose(
        -2, -1
    ).contiguous()


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


class PLCSObservationAugmentation(BaseObservationAugmentation[PLCSSample]):
    """Apply configured observation corruption to PLCS keypoint inputs."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(validate_augmentation(config))
        self.scale_human_uv = bool(self.uv_scale_cfg["apply_to_human"])
        self.scale_court_uv = bool(self.uv_scale_cfg["apply_to_court"])

    def _uv_scale_config(self) -> dict[str, Any]:
        return dict(cast(Mapping[str, Any], self.config["uv_scale"]))

    def _gaussian_config(self) -> dict[str, Any]:
        return dict(cast(Mapping[str, Any], self.config["gaussian_noise"]))

    def _visibility_dropout_config(self) -> dict[str, Any]:
        return dict(cast(Mapping[str, Any], self.config["visibility_dropout"]))

    def forward(self, sample: PLCSSample) -> PLCSSample:
        """Return an augmented PLCS sample."""
        return self.forward_with_tracking_provenance(sample).sample

    def forward_with_tracking_provenance(
        self,
        sample: PLCSSample,
    ) -> PLCSObservationTrackingResult:
        """Return the augmented sample and human pre-false-positive visibility."""
        if not self.enabled:
            return PLCSObservationTrackingResult(
                sample=sample,
                human_visibility_before_false_positive=sample["human_vis"]
                .bool()
                .clone(),
            )

        out: PLCSSample = clone_tensor_dict(sample)
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

        human_visibility_before_false_positive = out["human_vis"].bool().clone()
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
        return PLCSObservationTrackingResult(
            sample=out,
            human_visibility_before_false_positive=(
                human_visibility_before_false_positive
            ),
        )

    def _apply_uv_scale(self, sample: PLCSSample) -> None:
        cfg = self.uv_scale_cfg
        if not self._active(cfg, sample["human_kp"]):
            return
        scale_min, scale_max = self._parse_scale_range(cfg)
        scale = (
            torch.rand((), device=sample["human_kp"].device).item()
            * (scale_max - scale_min)
            + scale_min
        )
        if abs(scale - 1.0) < 1e-8:
            return
        if self.scale_human_uv:
            sample["human_kp"], sample["human_vis"] = scale_uv_with_visibility(
                sample["human_kp"],
                sample["human_vis"],
                float(scale),
            )
        if self.scale_court_uv:
            sample["court_kp"], sample["court_vis"] = scale_uv_with_visibility(
                sample["court_kp"],
                sample["court_vis"],
                float(scale),
            )

    def _apply_gaussian_noise(self, sample: PLCSSample) -> None:
        cfg = self.gaussian_cfg
        if not self._active(cfg, sample["human_kp"]):
            return
        human_std = float(cfg["human_std"])
        court_std = float(cfg["court_std"])
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
        entity: Literal["human", "court"],
    ) -> Tensor:
        cfg = self.temporal_jitter_cfg
        if not self._active(cfg, keypoints):
            return keypoints
        jitter_std = float(_entity_setting(cfg, entity, "jitter_std"))
        drift_std = float(_entity_setting(cfg, entity, "drift_std"))
        if jitter_std <= 0 and drift_std <= 0:
            return keypoints
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = (
            _flatten_temporal_tracks(
                keypoints,
                visibility,
            )
        )
        flat_keypoints = add_temporally_correlated_jitter(
            flat_keypoints,
            flat_visibility,
            jitter_std=jitter_std,
            drift_std=drift_std,
            drift_decay=_float_object(cfg["drift_decay"]),
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
        entity: Literal["human", "court"],
    ) -> tuple[Tensor, Tensor]:
        cfg = self.speed_conditioned_cfg
        if not self._active(cfg, keypoints):
            return keypoints, visibility
        frame_prob = float(_entity_setting(cfg, entity, "frame_prob"))
        if frame_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = (
            _flatten_temporal_tracks(
                keypoints,
                visibility,
            )
        )
        flat_keypoints, flat_visibility = apply_speed_conditioned_localization_error(
            flat_keypoints,
            flat_visibility,
            prob=frame_prob,
            speed_threshold=float(_entity_setting(cfg, entity, "speed_threshold")),
            lag_overshoot_range=_entity_setting(cfg, entity, "lag_overshoot_range"),
            noise_std=float(_entity_setting(cfg, entity, "noise_std")),
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
        entity: Literal["human", "court"],
    ) -> tuple[Tensor, Tensor]:
        cfg = self.edge_degradation_cfg
        if not self._active(cfg, keypoints):
            return keypoints, visibility
        noise_std = float(_entity_setting(cfg, entity, "noise_std"))
        drop_prob = float(_entity_setting(cfg, entity, "drop_prob"))
        clip_out_prob = float(_entity_setting(cfg, entity, "clip_out_prob"))
        if noise_std <= 0 and drop_prob <= 0 and clip_out_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = (
            _flatten_temporal_tracks(
                keypoints,
                visibility,
            )
        )
        flat_keypoints, flat_visibility = apply_edge_aware_degradation(
            flat_keypoints,
            flat_visibility,
            edge_margin=_float_object(cfg["edge_margin"]),
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
        entity: Literal["human", "court"],
    ) -> Tensor:
        cfg = self.visibility_dropout_cfg
        if not self._active(cfg, visibility):
            return visibility
        drop_prob = float(_entity_setting(cfg, entity, "drop_prob"))
        return random_visibility_dropout(visibility, drop_prob)

    def _apply_burst_dropout(
        self,
        visibility: Tensor,
        *,
        entity: Literal["human", "court"],
    ) -> Tensor:
        cfg = self.burst_dropout_cfg
        if not self._active(cfg, visibility):
            return visibility
        track_prob = float(_entity_setting(cfg, entity, "track_prob"))
        if track_prob <= 0:
            return visibility
        flat_visibility = visibility.transpose(-2, -1).reshape(-1, visibility.shape[-2])
        flat_visibility = apply_burst_visibility_dropout(
            flat_visibility,
            prob=track_prob,
            min_len=_int_object(cfg["min_len"]),
            max_len=_int_object(cfg["max_len"]),
            max_bursts=_int_object(cfg["max_bursts"]),
        )
        return (
            flat_visibility.reshape(
                *visibility.shape[:-2], visibility.shape[-1], visibility.shape[-2]
            )
            .transpose(-2, -1)
            .contiguous()
        )

    def _apply_false_positive(
        self,
        keypoints: Tensor,
        visibility: Tensor,
        *,
        entity: Literal["human", "court"],
        dropped_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.false_positive_cfg
        if not self._active(cfg, keypoints):
            return keypoints, visibility
        absent_prob = float(_entity_setting(cfg, entity, "prob_absent"))
        after_dropout_prob = float(_entity_setting(cfg, entity, "prob_after_dropout"))
        if absent_prob <= 0 and after_dropout_prob <= 0:
            return keypoints, visibility
        flat_keypoints, flat_visibility, prefix_shape, time_len, num_tracks = (
            _flatten_temporal_tracks(
                keypoints,
                visibility,
            )
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
            after_dropout_window=int(
                _entity_setting(cfg, entity, "after_dropout_window")
            ),
        )
        return _restore_temporal_tracks(
            flat_keypoints,
            flat_visibility,
            prefix_shape=prefix_shape,
            time_len=time_len,
            num_tracks=num_tracks,
        )
