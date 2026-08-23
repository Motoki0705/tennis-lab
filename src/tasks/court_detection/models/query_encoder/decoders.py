"""Variant-local dense decoder protocol and ablation families."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

from torch import Tensor, nn
from torch.nn import functional as F

from src.tasks.court_detection.configuration import (
    CourtQueryDecoderConfig,
    CourtQueryDPTDecoderConfig,
    CourtQueryLinearDecoderConfig,
    CourtQueryProgressiveDecoderConfig,
)
from src.tasks.court_detection.models.query_encoder.contracts import CourtEncoderTap


class CourtQueryDenseDecoder(Protocol):
    output_channels: int

    def __call__(
        self,
        taps: tuple[CourtEncoderTap, ...],
        *,
        output_hw: tuple[int, int],
    ) -> Tensor: ...


def _validate_output_hw(output_hw: tuple[int, int]) -> None:
    if len(output_hw) != 2 or any(
        type(item) is not int or item <= 0 for item in output_hw
    ):
        raise ValueError("Dense decoder output_hw must contain positive integers.")


def _select_taps(
    taps: tuple[CourtEncoderTap, ...],
    expected: tuple[int, ...],
) -> tuple[CourtEncoderTap, ...]:
    by_index: Mapping[int, CourtEncoderTap] = {tap.layer_index: tap for tap in taps}
    if len(by_index) != len(taps):
        raise ValueError("Dense decoder received duplicate encoder taps.")
    missing = tuple(index for index in expected if index not in by_index)
    if missing:
        raise ValueError(f"Dense decoder is missing declared tap(s): {missing}.")
    return tuple(by_index[index] for index in expected)


def _tap_to_map(tap: CourtEncoderTap) -> Tensor:
    tokens = tap.patch_tokens
    grid_h, grid_w = tap.grid_hw
    if tokens.ndim != 3 or tokens.shape[1] != grid_h * grid_w:
        raise ValueError("Cannot map encoder tap whose token/grid contract disagrees.")
    return tokens.transpose(1, 2).reshape(
        tokens.shape[0],
        tokens.shape[2],
        grid_h,
        grid_w,
    )


class DepthwiseSeparableRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return cast(Tensor, self.block(value))


class CourtQueryLinearDecoder(nn.Module):
    def __init__(self, *, hidden_dim: int, config: CourtQueryLinearDecoderConfig) -> None:
        super().__init__()
        self.tap_indices = config.tap_indices
        self.output_channels = config.width
        self.projection = nn.Conv2d(hidden_dim, config.width, kernel_size=1)

    def forward(
        self,
        taps: tuple[CourtEncoderTap, ...],
        *,
        output_hw: tuple[int, int],
    ) -> Tensor:
        _validate_output_hw(output_hw)
        (tap,) = _select_taps(taps, self.tap_indices)
        projected = self.projection(_tap_to_map(tap))
        return F.interpolate(
            projected,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )


class CourtQueryProgressiveDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        config: CourtQueryProgressiveDecoderConfig,
    ) -> None:
        super().__init__()
        self.tap_indices = config.tap_indices
        self.output_channels = config.width
        self.projection = nn.Conv2d(hidden_dim, config.width, kernel_size=1)
        self.stages = nn.ModuleList(
            DepthwiseSeparableRefinement(config.width)
            for _ in range(config.stage_count)
        )

    def forward(
        self,
        taps: tuple[CourtEncoderTap, ...],
        *,
        output_hw: tuple[int, int],
    ) -> Tensor:
        _validate_output_hw(output_hw)
        (tap,) = _select_taps(taps, self.tap_indices)
        value = self.projection(_tap_to_map(tap))
        for stage in self.stages:
            value = F.interpolate(
                value,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            value = stage(value)
        return F.interpolate(
            value,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )


class CourtQueryDPTDecoder(nn.Module):
    def __init__(self, *, hidden_dim: int, config: CourtQueryDPTDecoderConfig) -> None:
        super().__init__()
        self.tap_indices = config.tap_indices
        self.reassemble_factors = config.reassemble_factors
        self.output_channels = config.width
        self.projections = nn.ModuleList(
            nn.Conv2d(hidden_dim, config.width, kernel_size=1)
            for _ in self.tap_indices
        )
        self.fusions = nn.ModuleList(
            DepthwiseSeparableRefinement(config.width)
            for _ in range(len(self.tap_indices) - 1)
        )
        self.deepest_refinement = DepthwiseSeparableRefinement(config.width)

    @staticmethod
    def _reassemble(value: Tensor, factor: float) -> Tensor:
        height = round(value.shape[-2] * factor)
        width = round(value.shape[-1] * factor)
        if height <= 0 or width <= 0:
            raise ValueError("DPT reassemble factor produced a non-positive shape.")
        return F.interpolate(
            value,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        taps: tuple[CourtEncoderTap, ...],
        *,
        output_hw: tuple[int, int],
    ) -> Tensor:
        _validate_output_hw(output_hw)
        selected = _select_taps(taps, self.tap_indices)
        maps = [
            self._reassemble(projection(_tap_to_map(tap)), factor)
            for tap, projection, factor in zip(
                selected,
                self.projections,
                self.reassemble_factors,
                strict=True,
            )
        ]
        value = self.deepest_refinement(maps[-1])
        for fusion, skip in zip(
            reversed(self.fusions),
            reversed(maps[:-1]),
            strict=True,
        ):
            value = F.interpolate(
                value,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            value = fusion(value + skip)
        return F.interpolate(
            value,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )


CourtQueryDenseDecoderModule = (
    CourtQueryLinearDecoder | CourtQueryProgressiveDecoder | CourtQueryDPTDecoder
)


def build_query_dense_decoder(
    *,
    hidden_dim: int,
    config: CourtQueryDecoderConfig,
) -> CourtQueryDenseDecoderModule:
    if isinstance(config, CourtQueryLinearDecoderConfig):
        return CourtQueryLinearDecoder(hidden_dim=hidden_dim, config=config)
    if isinstance(config, CourtQueryProgressiveDecoderConfig):
        return CourtQueryProgressiveDecoder(hidden_dim=hidden_dim, config=config)
    if isinstance(config, CourtQueryDPTDecoderConfig):
        return CourtQueryDPTDecoder(hidden_dim=hidden_dim, config=config)
    raise TypeError(f"Unsupported query decoder config: {type(config).__name__}.")


__all__ = [
    "CourtQueryDPTDecoder",
    "CourtQueryDenseDecoder",
    "CourtQueryLinearDecoder",
    "CourtQueryProgressiveDecoder",
    "DepthwiseSeparableRefinement",
    "build_query_dense_decoder",
]
