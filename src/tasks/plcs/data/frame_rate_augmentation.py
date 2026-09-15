"""Synchronized train-time frame-rate sampling for PLCS scenes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.tasks.base.data.scene_dataset import TemporalWindow


@dataclass(frozen=True, slots=True)
class FrameRateSamplingPlan:
    """One source window and its output timestamps in source-frame units."""

    source_window: TemporalWindow
    source_fps: float
    output_fps: float
    source_positions: tuple[float, ...]

    @property
    def output_frames(self) -> int:
        return len(self.source_positions)

    def _positions(self, tensor: Tensor) -> Tensor:
        return torch.tensor(
            self.source_positions,
            dtype=torch.float64,
            device=tensor.device,
        )

    def _validate_axis(self, tensor: Tensor, *, axis: int) -> int:
        resolved = axis if axis >= 0 else tensor.ndim + axis
        if resolved < 0 or resolved >= tensor.ndim:
            raise ValueError(
                f"Temporal axis {axis} is invalid for {tuple(tensor.shape)}."
            )
        expected = self.source_window.seq_len
        if tensor.shape[resolved] != expected:
            raise ValueError(
                "Temporal tensor length disagrees with frame-rate source window: "
                f"expected {expected}, got {tensor.shape[resolved]}."
            )
        return resolved

    def nearest(self, tensor: Tensor, *, axis: int) -> Tensor:
        """Sample categorical, boolean, or visibility data by nearest frame."""
        resolved = self._validate_axis(tensor, axis=axis)
        indices = self._positions(tensor).round().long()
        return tensor.index_select(resolved, indices)

    def linear(self, tensor: Tensor, *, axis: int) -> Tensor:
        """Linearly sample floating-point values along one temporal axis."""
        if not tensor.is_floating_point():
            raise TypeError("Linear frame-rate sampling requires a floating tensor.")
        resolved = self._validate_axis(tensor, axis=axis)
        positions = self._positions(tensor)
        lower = positions.floor().long()
        upper = positions.ceil().long().clamp_max(self.source_window.seq_len - 1)
        alpha = (positions - lower).to(dtype=tensor.dtype)
        shape = [1] * tensor.ndim
        shape[resolved] = self.output_frames
        alpha = alpha.reshape(shape)
        lower_values = tensor.index_select(resolved, lower)
        upper_values = tensor.index_select(resolved, upper)
        return torch.lerp(lower_values, upper_values, alpha)

    def heading(self, tensor: Tensor, *, axis: int) -> Tensor:
        """Interpolate ``(cos(yaw), sin(yaw))`` along the shortest arc."""
        if tensor.shape[-1] != 2 or not tensor.is_floating_point():
            raise ValueError("Heading tensors must have floating shape (...,2).")
        resolved = self._validate_axis(tensor, axis=axis)
        positions = self._positions(tensor)
        lower = positions.floor().long()
        upper = positions.ceil().long().clamp_max(self.source_window.seq_len - 1)
        alpha = (positions - lower).to(dtype=tensor.dtype)
        shape = [1] * (tensor.ndim - 1)
        shape[resolved] = self.output_frames
        alpha = alpha.reshape(shape)
        lower_values = tensor.index_select(resolved, lower)
        upper_values = tensor.index_select(resolved, upper)
        lower_angle = torch.atan2(lower_values[..., 1], lower_values[..., 0])
        upper_angle = torch.atan2(upper_values[..., 1], upper_values[..., 0])
        delta = torch.atan2(
            torch.sin(upper_angle - lower_angle),
            torch.cos(upper_angle - lower_angle),
        )
        angle = lower_angle + alpha * delta
        return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)

    @staticmethod
    def _broadcast_validity(validity: Tensor, *, target_ndim: int) -> Tensor:
        return validity.reshape(
            (*validity.shape, *((1,) * (target_ndim - validity.ndim)))
        )

    def masked_linear(
        self,
        tensor: Tensor,
        validity: Tensor,
        *,
        axis: int,
    ) -> tuple[Tensor, Tensor]:
        """Interpolate continuous values without blending through invalid zeros.

        ``validity`` must describe a prefix of ``tensor`` (for example
        ``(T,P)`` for ``(T,P,17,3)``). Where both interpolation endpoints are
        valid, normal linear interpolation is used. At a validity boundary the
        nearest endpoint is copied; an invalid nearest endpoint yields zero.
        """
        resolved, validity_bool = self._validate_masked_inputs(
            tensor, validity, axis=axis
        )
        positions = self._positions(tensor)
        lower = positions.floor().long()
        upper = positions.ceil().long().clamp_max(self.source_window.seq_len - 1)
        nearest = positions.round().long()
        interpolated = self.linear(tensor, axis=resolved)
        nearest_values = tensor.index_select(resolved, nearest)
        lower_valid = validity_bool.index_select(resolved, lower)
        upper_valid = validity_bool.index_select(resolved, upper)
        output_valid_bool = validity_bool.index_select(resolved, nearest)
        both_valid = self._broadcast_validity(
            lower_valid & upper_valid,
            target_ndim=tensor.ndim,
        )
        output_valid_broadcast = self._broadcast_validity(
            output_valid_bool,
            target_ndim=tensor.ndim,
        )
        values = torch.where(both_valid, interpolated, nearest_values)
        output_valid = (
            output_valid_bool
            if validity.dtype == torch.bool
            else output_valid_bool.to(dtype=validity.dtype)
        )
        return torch.where(output_valid_broadcast, values, 0.0), output_valid

    def masked_heading(
        self,
        tensor: Tensor,
        validity: Tensor,
        *,
        axis: int,
    ) -> tuple[Tensor, Tensor]:
        """Interpolate headings without crossing absent lifecycle frames."""
        resolved, validity_bool = self._validate_masked_inputs(
            tensor, validity, axis=axis
        )
        positions = self._positions(tensor)
        lower = positions.floor().long()
        upper = positions.ceil().long().clamp_max(self.source_window.seq_len - 1)
        nearest = positions.round().long()
        interpolated = self.heading(tensor, axis=resolved)
        nearest_values = tensor.index_select(resolved, nearest)
        lower_valid = validity_bool.index_select(resolved, lower)
        upper_valid = validity_bool.index_select(resolved, upper)
        output_valid_bool = validity_bool.index_select(resolved, nearest)
        both_valid = self._broadcast_validity(
            lower_valid & upper_valid,
            target_ndim=tensor.ndim,
        )
        output_valid_broadcast = self._broadcast_validity(
            output_valid_bool,
            target_ndim=tensor.ndim,
        )
        values = torch.where(both_valid, interpolated, nearest_values)
        identity = torch.zeros_like(values)
        identity[..., 0] = 1.0
        output_valid = (
            output_valid_bool
            if validity.dtype == torch.bool
            else output_valid_bool.to(dtype=validity.dtype)
        )
        return torch.where(output_valid_broadcast, values, identity), output_valid

    def _validate_masked_inputs(
        self,
        tensor: Tensor,
        validity: Tensor,
        *,
        axis: int,
    ) -> tuple[int, Tensor]:
        if not tensor.is_floating_point():
            raise TypeError("Masked interpolation requires a floating tensor.")
        if validity.dtype == torch.bool:
            validity_bool = validity
        elif validity.is_floating_point():
            if not bool(((validity == 0.0) | (validity == 1.0)).all()):
                raise ValueError(
                    "Floating masked-interpolation validity must be binary."
                )
            validity_bool = validity.bool()
        else:
            raise TypeError(
                "Masked interpolation validity must be bool or floating binary."
            )
        if validity.ndim > tensor.ndim or tuple(validity.shape) != tuple(
            tensor.shape[: validity.ndim]
        ):
            raise ValueError(
                "Masked interpolation validity must match a tensor-shape prefix."
            )
        resolved = self._validate_axis(tensor, axis=axis)
        self._validate_axis(validity, axis=resolved)
        return resolved, validity_bool


class PLCSFrameRateSampler:
    """Choose a train-time FPS and produce one synchronized sampling plan."""

    def __init__(self, augmentation_config: Mapping[str, Any]) -> None:
        augmentation = as_config_mapping(
            augmentation_config,
            path="data.augmentation",
        )
        self.augmentation_enabled = cast(
            bool,
            require_config_value(
                augmentation,
                "enabled",
                bool,
                path="data.augmentation",
            ),
        )
        block = require_config_mapping(
            augmentation,
            "frame_rate",
            path="data.augmentation",
        )
        self.enabled = cast(
            bool,
            require_config_value(
                block,
                "enabled",
                bool,
                path="data.augmentation.frame_rate",
            ),
        )
        self.probability = float(
            cast(
                "float | int",
                require_config_value(
                    block,
                    "prob",
                    (float, int),
                    path="data.augmentation.frame_rate",
                ),
            )
        )
        raw_choices = require_config_value(
            block,
            "choices_hz",
            (list, tuple),
            path="data.augmentation.frame_rate",
        )
        choice_values = tuple(cast(Sequence[object], raw_choices))
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in choice_values
        ):
            raise TypeError("data.augmentation.frame_rate.choices_hz must be numeric.")
        choices = tuple(float(cast("float | int", value)) for value in choice_values)
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("data.augmentation.frame_rate.prob must be within [0,1].")
        if not choices or any(
            not math.isfinite(value) or value <= 0.0 for value in choices
        ):
            raise ValueError(
                "data.augmentation.frame_rate.choices_hz must contain positive finite values."
            )
        if len(set(choices)) != len(choices):
            raise ValueError("data.augmentation.frame_rate.choices_hz must be unique.")
        self.choices_hz = choices

    def plan(
        self,
        *,
        source_fps: object,
        full_len: int,
        seq_len_range: tuple[int, int],
        crop_mode: Literal["random", "center"],
        augment: bool,
        rng: np.random.Generator,
    ) -> FrameRateSamplingPlan:
        """Plan a fixed-capacity output sampled from a sufficiently wide crop."""
        if isinstance(source_fps, bool) or not isinstance(source_fps, (int, float)):
            raise TypeError("Scene meta.fps must be numeric.")
        native = float(source_fps)
        if not math.isfinite(native) or native <= 0.0:
            raise ValueError("Scene meta.fps must be finite and positive.")
        if type(full_len) is not int or full_len <= 0:
            raise ValueError("full_len must be a positive integer.")
        min_frames, max_frames = seq_len_range
        if min_frames <= 0 or max_frames < min_frames:
            raise ValueError("seq_len_range must be a positive ordered pair.")
        if crop_mode not in {"random", "center"}:
            raise ValueError("crop_mode must be random or center.")

        active = self.augmentation_enabled and self.enabled and augment
        if active and self.probability < 1.0:
            active = bool(rng.random() < self.probability)
        candidates = self.choices_hz if active else (native,)
        feasible = tuple(
            candidate
            for candidate in candidates
            if self._maximum_output_frames(
                full_len=full_len,
                source_fps=native,
                output_fps=candidate,
            )
            >= min_frames
        )
        if not feasible:
            raise ValueError(
                "Scene is too short for every configured frame-rate choice: "
                f"frames={full_len}, native_fps={native}, choices={candidates}, "
                f"minimum_output_frames={min_frames}."
            )
        output_fps = float(feasible[int(rng.integers(0, len(feasible)))])
        capacity = self._maximum_output_frames(
            full_len=full_len,
            source_fps=native,
            output_fps=output_fps,
        )
        output_frames = int(rng.integers(min_frames, min(max_frames, capacity) + 1))
        source_step = native / output_fps
        positions = np.arange(output_frames, dtype=np.float64) * source_step
        source_frames = int(math.ceil(float(positions[-1]))) + 1
        max_start = full_len - source_frames
        start = (
            int(rng.integers(0, max_start + 1))
            if crop_mode == "random" and max_start > 0
            else max_start // 2
        )
        window = TemporalWindow(
            start=start,
            end=start + source_frames,
            seq_len=source_frames,
            full_len=full_len,
        )
        return FrameRateSamplingPlan(
            source_window=window,
            source_fps=native,
            output_fps=output_fps,
            source_positions=tuple(float(value) for value in positions),
        )

    @staticmethod
    def _maximum_output_frames(
        *,
        full_len: int,
        source_fps: float,
        output_fps: float,
    ) -> int:
        return int(math.floor((full_len - 1) * output_fps / source_fps)) + 1


__all__ = ["FrameRateSamplingPlan", "PLCSFrameRateSampler"]
