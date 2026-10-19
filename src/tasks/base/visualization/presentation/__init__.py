"""Shared primitives for presentation timelines, diagrams and raster exports."""

from src.tasks.base.visualization.presentation.architecture import (
    ArchitectureState,
    Block,
    BlockState,
    Edge,
    draw_architecture,
    draw_flow_particles,
)
from src.tasks.base.visualization.presentation.curves import (
    CurveSeries,
    draw_progressive_curves,
)
from src.tasks.base.visualization.presentation.export import render_presentation
from src.tasks.base.visualization.presentation.timeline import Segment, StepClock

__all__ = [
    "ArchitectureState",
    "Block",
    "BlockState",
    "CurveSeries",
    "Edge",
    "Segment",
    "StepClock",
    "draw_architecture",
    "draw_flow_particles",
    "draw_progressive_curves",
    "render_presentation",
]
