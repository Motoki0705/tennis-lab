"""Coordinate and per-frame event heads."""

from torch import nn

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES
from src.utils.models.components import RMSNorm


def trajectory_head(width: int) -> nn.Sequential:
    return nn.Sequential(RMSNorm(width), nn.Linear(width, 3))


def event_head(width: int) -> nn.Sequential:
    return nn.Sequential(RMSNorm(width), nn.Linear(width, len(EVENT_CLASSES)))
