"""Coordinate and per-frame binary event heads."""

from torch import nn

from src.utils.models.components import RMSNorm


def trajectory_head(width: int) -> nn.Sequential:
    return nn.Sequential(RMSNorm(width), nn.Linear(width, 3))


def event_head(width: int) -> nn.Sequential:
    return nn.Sequential(RMSNorm(width), nn.Linear(width, 2))
