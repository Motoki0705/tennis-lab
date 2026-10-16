"""Versioned five-channel input; the historical two-channel contract stays intact."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .mdd import RGBToMDD


class RGBToRGBMDD(RGBToMDD):
    """The same sampled, augmented RGB supplies appearance and fixed FP32 motion."""

    def input_contract(self) -> dict[str, Any]:
        return dict(
            super().input_contract(),
            schema="rgb_uint8_native_rgb_mdd.v1",
            frames=32,
            encoder_layout="B,5,T,H,W",
            encoder_channels=["R", "G", "B", "MDD_positive", "MDD_negative"],
            encoder_value_range=[0, 1],
            rgb_normalization="divide_255_only",
            temporal_difference="adjacent_sampled_frames",
            spatial_alignment="same_augmented_rgb_as_gt",
        )

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 5 or rgb.shape[1] != 32:
            raise ValueError("RGB+MDD requires exactly 32 sampled frames")
        mdd = super().forward(rgb)
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            pixels = rgb.to(torch.float32).transpose(1, 2) / 255
            return torch.cat((pixels, mdd), dim=1)


def build_input_transform(input_channels: int, *, a: float = .2, b: float = .15) -> RGBToMDD:
    if type(input_channels) is not int or input_channels not in (2, 5):
        raise ValueError("Input channels must explicitly select 2 (MDD) or 5 (RGB+MDD)")
    return RGBToRGBMDD(a, b) if input_channels == 5 else RGBToMDD(a, b)


def input_transform_from_contract(contract: dict[str, Any], *, input_channels: int) -> RGBToMDD:
    """Reject mismatched normalization, channel order, time semantics and topology."""
    if not isinstance(contract, dict) or "mdd_a" not in contract or "mdd_b" not in contract:
        raise ValueError("Missing explicit RGB/MDD input contract")
    transform = build_input_transform(input_channels, a=contract["mdd_a"], b=contract["mdd_b"])
    if contract != transform.input_contract():
        raise ValueError("RGB/MDD input contract mismatch with configured input channels")
    return transform
