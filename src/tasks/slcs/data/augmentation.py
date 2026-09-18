"""Detector failures and modality dropout, applied to SLCS inputs only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from src.tasks.slcs.data.types import SLCSSample

INPUT_CONDITIONS = frozenset({"full", "no_rgb", "rgb_only", "detector_gap"})


@dataclass(frozen=True)
class ObservationAugmentationConfig:
    enabled: bool
    joint_dropout: float
    ball_dropout: float
    uv_std: float
    burst_probability: float
    burst_max_frames: int
    rgb_only_probability: float
    rgb_dropout_probability: float

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.burst_max_frames) is not int
            or self.burst_max_frames < 1
        ):
            raise ValueError(
                "Augmentation needs boolean enabled and positive integer burst_max_frames"
            )
        for name in (
            "joint_dropout",
            "ball_dropout",
            "burst_probability",
            "rgb_only_probability",
            "rgb_dropout_probability",
        ):
            if (
                not math.isfinite(getattr(self, name))
                or not 0 <= getattr(self, name) <= 1
            ):
                raise ValueError(f"augmentation.{name} must be in [0,1]")
        if not math.isfinite(self.uv_std) or self.uv_std < 0:
            raise ValueError("augmentation.uv_std must be finite and nonnegative")


def augment_observations(
    sample: SLCSSample, config: ObservationAugmentationConfig
) -> SLCSSample:
    """Use worker-seeded torch RNG; never modify targets or cached source arrays."""
    if not config.enabled:
        return sample
    inputs = {
        "player_kp",
        "player_kp_vis",
        "player_valid",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "dino_tokens",
        "dino_padding_mask",
    }
    out = {
        key: tensor.clone() if key in inputs else tensor
        for key, tensor in cast(dict[str, torch.Tensor], sample).items()
    }
    rgb_only = bool(torch.rand(()) < config.rgb_only_probability)
    if rgb_only:
        if out["dino_padding_mask"].all():
            raise ValueError("RGB-only augmentation requires observed DINO tokens")
        for name in (
            "player_kp",
            "player_kp_vis",
            "player_valid",
            "ball_uv",
            "ball_vis",
            "court_kp",
            "court_vis",
        ):
            out[name].zero_()
    else:
        out["player_kp_vis"] *= (
            torch.rand_like(out["player_kp_vis"]) >= config.joint_dropout
        )
        out["ball_vis"] &= torch.rand(out["ball_vis"].shape) >= config.ball_dropout
        length = int((~out["padding_mask"]).sum())
        if length and bool(torch.rand(()) < config.burst_probability):
            size = int(
                torch.randint(1, min(config.burst_max_frames, length) + 1, ()).item()
            )
            start = int(torch.randint(0, length - size + 1, ()).item())
            out["ball_vis"][start : start + size] = False
            player = int(torch.randint(0, out["player_kp_vis"].shape[0], ()).item())
            out["player_kp_vis"][player, start : start + size] = 0
        out["player_valid"] &= out["player_kp_vis"].amax(-1) > 0
        for coordinates, visibility, noise_scale in (
            ("player_kp", "player_kp_vis", 1.0),
            ("ball_uv", "ball_vis", 1.0),
            ("court_kp", "court_vis", 0.25),
        ):
            value = out[coordinates]
            noise = torch.randn_like(value) * config.uv_std * noise_scale
            out[coordinates] = torch.where(
                out[visibility][..., None] > 0, (value + noise).clamp(0, 1), 0
            )
        if bool(torch.rand(()) < config.rgb_dropout_probability):
            out["dino_tokens"].zero_()
            out["dino_padding_mask"].fill_(True)
    return cast(SLCSSample, out)
