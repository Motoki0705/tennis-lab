"""Task-specific heads for ball multi-task models."""

from src.ball_multitask.models.heads.trajectory_head import Trajectory3DHeadAdapter
from src.ball_multitask.models.heads.uv_head import UVCompletionHead
from src.ball_multitask.models.heads.event_head import EventLogitsHeadAdapter

__all__ = ["UVCompletionHead", "Trajectory3DHeadAdapter", "EventLogitsHeadAdapter"]
